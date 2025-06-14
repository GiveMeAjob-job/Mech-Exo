#!/bin/bash
# Network Chaos Injection using Linux tc/netem
# Applies network delay and packet loss to Kubernetes pods

set -euo pipefail

# Default values
DELAY="200ms"
LOSS="2%"
JITTER="10ms"
DURATION="300"  # 5 minutes
APP_LABEL="app=mech-exo-exec"
NAMESPACE="default"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --delay)
            DELAY="$2"
            shift 2
            ;;
        --loss)
            LOSS="$2"
            shift 2
            ;;
        --jitter)
            JITTER="$2"
            shift 2
            ;;
        --duration)
            DURATION="$2"
            shift 2
            ;;
        --app)
            APP_LABEL="$2"
            shift 2
            ;;
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --help)
            cat << EOF
Network Chaos Injection Tool

Usage: $0 [OPTIONS]

Options:
    --delay DELAY       Network delay (default: 200ms)
    --loss LOSS         Packet loss percentage (default: 2%)
    --jitter JITTER     Delay jitter (default: 10ms)
    --duration SECONDS  Duration in seconds (default: 300)
    --app LABEL         Kubernetes app label selector (default: app=mech-exo-exec)
    --namespace NS      Kubernetes namespace (default: default)
    --help              Show this help message

Examples:
    $0 --delay 100ms --loss 1%
    $0 --delay 500ms --loss 5% --duration 600
    $0 --app app=api-server --namespace production

EOF
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

log_info "Starting network chaos injection..."
log_info "Target: ${APP_LABEL} in namespace ${NAMESPACE}"
log_info "Parameters: delay=${DELAY}, loss=${LOSS}, jitter=${JITTER}, duration=${DURATION}s"

# Get list of target pods
log_info "Finding target pods..."
PODS=$(kubectl get pods -n "${NAMESPACE}" -l "${APP_LABEL}" -o jsonpath='{.items[*].metadata.name}')

if [ -z "$PODS" ]; then
    log_error "No pods found with label ${APP_LABEL} in namespace ${NAMESPACE}"
    exit 1
fi

POD_ARRAY=($PODS)
log_success "Found ${#POD_ARRAY[@]} target pods: ${PODS}"

# Function to apply network chaos to a pod
apply_chaos() {
    local pod_name=$1
    log_info "Applying network chaos to pod: ${pod_name}"
    
    # Check if pod is ready
    if ! kubectl get pod "${pod_name}" -n "${NAMESPACE}" -o jsonpath='{.status.phase}' | grep -q "Running"; then
        log_warning "Pod ${pod_name} is not running, skipping..."
        return 1
    fi
    
    # Apply traffic control rules
    if kubectl exec -n "${NAMESPACE}" "${pod_name}" -- \
        tc qdisc add dev eth0 root netem delay "${DELAY}" "${JITTER}" loss "${LOSS}" 2>/dev/null; then
        log_success "Network chaos applied to ${pod_name}"
        echo "${pod_name}" >> /tmp/chaos_pods.txt
        return 0
    else
        # Try alternative interface names
        for iface in eth0 ens3 ens4 veth0; do
            if kubectl exec -n "${NAMESPACE}" "${pod_name}" -- \
                tc qdisc add dev "${iface}" root netem delay "${DELAY}" "${JITTER}" loss "${LOSS}" 2>/dev/null; then
                log_success "Network chaos applied to ${pod_name} (interface: ${iface})"
                echo "${pod_name}:${iface}" >> /tmp/chaos_pods.txt
                return 0
            fi
        done
        
        log_warning "Failed to apply network chaos to ${pod_name} - no suitable network interface found"
        return 1
    fi
}

# Function to remove network chaos from a pod
remove_chaos() {
    local pod_info=$1
    local pod_name=$(echo "${pod_info}" | cut -d: -f1)
    local interface=$(echo "${pod_info}" | cut -d: -f2)
    
    # Use default interface if not specified
    if [ "${interface}" = "${pod_name}" ]; then
        interface="eth0"
    fi
    
    log_info "Removing network chaos from pod: ${pod_name} (interface: ${interface})"
    
    if kubectl exec -n "${NAMESPACE}" "${pod_name}" -- \
        tc qdisc del dev "${interface}" root 2>/dev/null; then
        log_success "Network chaos removed from ${pod_name}"
    else
        log_warning "Failed to remove network chaos from ${pod_name} (may have been cleaned up already)"
    fi
}

# Function to handle cleanup on exit
cleanup() {
    log_info "Cleaning up network chaos..."
    
    if [ -f /tmp/chaos_pods.txt ]; then
        while IFS= read -r pod_info; do
            if [ -n "${pod_info}" ]; then
                remove_chaos "${pod_info}"
            fi
        done < /tmp/chaos_pods.txt
        rm -f /tmp/chaos_pods.txt
    fi
    
    log_success "Cleanup completed"
}

# Set up cleanup trap
trap cleanup EXIT INT TERM

# Clear previous chaos pods file
rm -f /tmp/chaos_pods.txt

# Apply chaos to all target pods
log_info "Applying network chaos to all target pods..."
success_count=0
fail_count=0

for pod in "${POD_ARRAY[@]}"; do
    if apply_chaos "${pod}"; then
        ((success_count++))
    else
        ((fail_count++))
    fi
done

log_info "Chaos application complete: ${success_count} successful, ${fail_count} failed"

if [ ${success_count} -eq 0 ]; then
    log_error "No pods affected by network chaos"
    exit 1
fi

# Wait for specified duration
log_info "Network chaos active for ${DURATION} seconds..."
log_info "You can monitor the effects with: kubectl logs -f -l ${APP_LABEL} -n ${NAMESPACE}"

# Create a background process to log network statistics
{
    while [ -f /tmp/chaos_pods.txt ]; do
        echo "$(date): Network chaos still active on $(wc -l < /tmp/chaos_pods.txt) pods"
        sleep 60
    done
} &
STATS_PID=$!

# Wait for the specified duration
sleep "${DURATION}"

# Kill the stats process
kill ${STATS_PID} 2>/dev/null || true

log_success "Network chaos duration completed"

# Cleanup is handled by the trap
log_info "Network chaos injection completed successfully"

# Generate report
cat > /tmp/chaos_logs/network_chaos_report.txt << EOF
Network Chaos Injection Report
==============================
Timestamp: $(date -u)
Target: ${APP_LABEL} in namespace ${NAMESPACE}
Parameters:
  - Delay: ${DELAY}
  - Packet Loss: ${LOSS}
  - Jitter: ${JITTER}
  - Duration: ${DURATION} seconds

Results:
  - Pods targeted: ${#POD_ARRAY[@]}
  - Successfully affected: ${success_count}
  - Failed: ${fail_count}
  - Success rate: $(( success_count * 100 / ${#POD_ARRAY[@]} ))%

Affected pods:
$(cat /tmp/chaos_pods.txt 2>/dev/null || echo "None")

Notes:
- Network chaos simulates real-world network conditions
- Monitor application metrics during this period
- Cleanup is automatic after duration expires
EOF

log_success "Network chaos report generated: /tmp/chaos_logs/network_chaos_report.txt"