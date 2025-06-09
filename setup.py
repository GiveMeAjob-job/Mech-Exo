from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open('requirements.txt', 'r') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Dash-specific requirements
dash_requirements = [
    'dash>=2.14.0',
    'plotly>=5.17.0', 
    'jinja2>=3.1.0',
    'gunicorn>=21.2.0',
    'flask-httpauth>=4.8.0'
]

setup(
    name="mech-exo",
    version="0.4.0",
    description="Mechanical Exoskeleton - Systematic Trading System",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        'dash': dash_requirements,
        'dev': [
            'pytest>=7.4.0',
            'pytest-cov>=4.1.0',
            'black>=23.9.0',
            'ruff>=0.1.0',
            'jupyter>=1.0.0'
        ]
    },
    entry_points={
        'console_scripts': [
            'exo=mech_exo.cli:main',
        ],
    },
    python_requires='>=3.11',
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
    ],
)