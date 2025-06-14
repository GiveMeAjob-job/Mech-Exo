# Incident Postmortem Template

**Incident ID**: [INCIDENT-YYYY-MM-DD-###]  
**Date**: [YYYY-MM-DD]  
**Reporter**: [Name]  
**Incident Commander**: [Name]  
**Severity**: [Critical/High/Medium/Low]

---

## 📋 Executive Summary

<!-- Brief summary of the incident for executives and stakeholders -->

**What happened**: [One-sentence description]  
**Impact**: [Business impact - trading downtime, financial loss, etc.]  
**Duration**: [Total incident duration]  
**Root Cause**: [Primary root cause in simple terms]  
**Resolution**: [How it was fixed]

---

## 🔍 Incident Details

### Timeline of Events

| Time (UTC) | Event | Action Taken | Person |
|------------|-------|--------------|--------|
| HH:MM | [Initial symptom/alert] | [First response action] | [Name] |
| HH:MM | [Escalation/discovery] | [Investigation action] | [Name] |
| HH:MM | [Root cause identified] | [Mitigation action] | [Name] |
| HH:MM | [Resolution] | [Fix implemented] | [Name] |
| HH:MM | [Post-incident] | [Monitoring/validation] | [Name] |

### Impact Assessment

**Systems Affected**:
- [ ] Trading Engine
- [ ] Risk Management 
- [ ] Data Pipeline
- [ ] Dashboard/UI
- [ ] Alerting System
- [ ] External APIs

**Business Impact**:
- **Trading Halt Duration**: [X minutes]
- **Financial Impact**: [$ amount or N/A]
- **Positions Affected**: [Number of positions/symbols]
- **Customer Impact**: [Internal teams affected]
- **Data Loss**: [Any data lost/corrupted]

**Technical Impact**:
- **Availability**: [% uptime during incident]
- **Performance**: [Response time degradation]
- **Data Integrity**: [Any data consistency issues]
- **Alert Storm**: [Number of alerts generated]

---

## 🔧 Technical Analysis

### Root Cause Analysis

**Primary Root Cause**:
[Detailed technical explanation of what went wrong]

**Contributing Factors**:
1. [Factor 1 - e.g., configuration error]
2. [Factor 2 - e.g., monitoring gap]
3. [Factor 3 - e.g., process failure]

**Why did it happen?** (5 Whys Analysis):
1. Why did [incident] occur? → [Answer 1]
2. Why did [Answer 1] happen? → [Answer 2]
3. Why did [Answer 2] happen? → [Answer 3]
4. Why did [Answer 3] happen? → [Answer 4]
5. Why did [Answer 4] happen? → [Answer 5 - Root cause]

### Detection and Response

**How was it detected?**
- [ ] Automated alert
- [ ] User report
- [ ] Monitoring dashboard
- [ ] Health check failure
- [ ] Manual discovery

**Time to Detection**: [X minutes from start]  
**Time to Mitigation**: [X minutes from detection]  
**Time to Resolution**: [X minutes from mitigation]

**What went well?**
- [Positive aspects of the response]
- [Effective procedures that worked]
- [Good communication/coordination]

**What could be improved?**
- [Detection delays]
- [Response inefficiencies]
- [Communication gaps]

---

## 📊 Monitoring and Alerting

### Alert Performance

**Alerts that fired**:
- [Alert name] - [Time] - [Effectiveness: High/Medium/Low]
- [Alert name] - [Time] - [Effectiveness: High/Medium/Low]

**Alerts that should have fired but didn't**:
- [Missing alert] - [Why it didn't fire]

**False positives during incident**:
- [Alert name] - [Why it was false positive]

### Monitoring Gaps

**What metrics should we have been monitoring?**
- [Missing metric 1]
- [Missing metric 2]

**Dashboard improvements needed**:
- [Dashboard enhancement 1]
- [Dashboard enhancement 2]

---

## 🛠️ Action Items

### Immediate Actions (Complete within 24 hours)
- [ ] **[Action]** - Owner: [Name] - Due: [Date]
- [ ] **[Action]** - Owner: [Name] - Due: [Date]

### Short-term Actions (Complete within 1 week)
- [ ] **[Action]** - Owner: [Name] - Due: [Date]
- [ ] **[Action]** - Owner: [Name] - Due: [Date]

### Long-term Actions (Complete within 1 month)
- [ ] **[Action]** - Owner: [Name] - Due: [Date]
- [ ] **[Action]** - Owner: [Name] - Due: [Date]

### Process Improvements
- [ ] **[Process change]** - Owner: [Name] - Due: [Date]
- [ ] **[Documentation update]** - Owner: [Name] - Due: [Date]

---

## 📚 Lessons Learned

### What We Learned
1. **[Lesson 1]** - [Explanation]
2. **[Lesson 2]** - [Explanation]
3. **[Lesson 3]** - [Explanation]

### Best Practices to Adopt
- [Best practice 1]
- [Best practice 2]

### Anti-patterns to Avoid
- [Anti-pattern 1]
- [Anti-pattern 2]

---

## 🔄 Prevention Measures

### Technical Improvements
- **Monitoring**: [New alerts/dashboards to implement]
- **Testing**: [Additional tests to prevent recurrence]
- **Infrastructure**: [System improvements needed]
- **Code**: [Code changes required]

### Process Improvements
- **Documentation**: [Runbooks to create/update]
- **Training**: [Team training needs]
- **Procedures**: [Process changes needed]
- **Communication**: [Communication improvements]

---

## 📎 Supporting Information

### Logs and Evidence
- [Link to relevant log files]
- [Screenshots of monitoring dashboards]
- [Error messages and stack traces]
- [Configuration files involved]

### Related Incidents
- [Link to similar past incidents]
- [Related known issues]

### External Dependencies
- [Third-party services involved]
- [External factors that contributed]

---

## 👥 Incident Response Team

**Incident Commander**: [Name] - [Role]  
**Technical Lead**: [Name] - [Role]  
**Communications Lead**: [Name] - [Role]  
**Additional Responders**: 
- [Name] - [Role] - [Contribution]
- [Name] - [Role] - [Contribution]

---

## 📈 Metrics

### Incident Response Metrics
- **MTTD** (Mean Time to Detection): [X minutes]
- **MTTR** (Mean Time to Resolution): [X minutes]
- **MTBF** (Mean Time Between Failures): [X days]
- **Availability Impact**: [X.XX% of monthly SLA]

### Business Metrics
- **Trading Downtime**: [X minutes]
- **Positions Affected**: [X positions]
- **Revenue Impact**: [$ amount]
- **Customer Satisfaction**: [Survey results if applicable]

---

## ✅ Postmortem Review

**Postmortem Review Date**: [Date]  
**Attendees**: [List of attendees]  
**Review Status**: [Complete/In Progress/Scheduled]

**Action Item Status** (as of [Date]):
- Immediate Actions: [X/Y complete]
- Short-term Actions: [X/Y complete]  
- Long-term Actions: [X/Y complete]

---

## 📝 Follow-up

**Follow-up Review Date**: [Date + 1 month]  
**Owner**: [Name]  
**Success Criteria**: 
- [ ] All action items completed
- [ ] No recurrence of similar incident
- [ ] Improved detection/response times
- [ ] Updated documentation in place

---

*This postmortem follows the [Blameless Postmortem](https://sre.google/sre-book/postmortem-culture/) principles. The goal is learning and improvement, not blame.*

**Template Version**: 1.0  
**Last Updated**: 2024-12-13  
**Owner**: Trading Operations Team