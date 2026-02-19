# Nuclear Power Plant Simulation

**Version 1.0**

A comprehensive real-time simulation of a nuclear power plant (NPP) using Python. This project models the complete thermodynamic cycle of a pressurized water reactor (PWR) with neutron transport capabilities using OpenMC, real-time control systems, telemetry logging, and machine learning-based prediction engines.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Components](#components)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Simulation](#running-the-simulation)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Key Features](#key-features)
- [Data](#data)
- [Technology Stack](#technology-stack)

---

## Overview

This simulation framework provides a full-scale digital twin of a nuclear power plant. It combines:

1. **Neutron Transport Simulation** - Using OpenMC for Monte Carlo neutron transport and criticality calculations
2. **Thermal-Hydraulic Modeling** - Primary and secondary coolant loops with heat transfer
3. **Control System Simulation** - Real-time control rod movement, scram procedures, and safety systems
4. **Turbine/Generator Systems** - Steam turbine and electrical generator modeling
5. **Grid Interface** - Power output integration with electrical grids
6. **Telemetry & Logging** - Real-time data collection and monitoring
7. **ML Prediction Engine** - Machine learning models for predictive maintenance and anomaly detection

---

## Architecture

The system uses a **distributed actor-style architecture** with ZeroMQ (ZMQ) for inter-process communication. Each component runs as an independent process, communicating through IPC (inter-process communication) channels.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Control System                               │
│         (Master clock, control rods, safety systems)               │
└─────────────────────────────────────────────────────────────────────┘
            │                    │                    │
            ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  Reactor Core  │  │  Coolant Loop   │  │   Turbine      │
│  (OpenMC)      │  │  (Primary)      │  │   Generator    │
└─────────────────┘  └─────────────────┘  └─────────────────┘
            │                    │                    │
            ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ Steam Generator│  │  Condenser      │  │   Grid         │
│                 │  │  Cooling        │  │   Interface    │
└─────────────────┘  └─────────────────┘  └─────────────────┘
            │                    │
            ▼                    ▼
      ┌─────────────────┐  ┌─────────────────┐
      │   Dashboard    │  │  Telemetry      │
      │   (Monitoring) │  │  Logging        │
      └─────────────────┘  └─────────────────┘
```

### Communication Pattern

- **PUB/SUB**: Telemetry broadcasting (tick signals, heartbeats)
- **DEALER/ROUTER**: Request-response control commands
- **PUSH/PULL**: Data streaming between components

---

## Components

### 1. Reactor Core (`components/reactor_core/`)
The heart of the simulation - models a PWR reactor with:
- Point kinetics neutron transport
- Delayed neutron precursor dynamics
- Fuel, cladding, and moderator temperature calculations
- Control rod bank management (6 banks: A-F)
- Reactivity feedback coefficients
- Burnup and fuel management

### 2. Primary Coolant Loop (`components/primary_coolant_loop/`)
Models the primary cooling system:
- Coolant flow rate and pressure
- Heat transfer from reactor to steam generators
- Temperature distribution (inlet/outlet)

### 3. Steam Generator (`components/steam_generator/`)
Secondary loop heat exchanger:
- Steam generation from primary coolant heat
- Steam pressure and temperature modeling

### 4. Turbine Generator (`components/turbine_generator/`)
Steam turbine and electrical generator:
- Turbine efficiency modeling
- Power output calculation

### 5. Condenser/Cooling (`components/condenser_cooling/`)
Heat rejection system:
- Condenser cooling water source
- Cooling tower integration

### 6. Control System (`components/control_safety/`)
Safety and control systems:
- Control rod insertion/withdrawal
- SCRAM (emergency shutdown) procedures
- Boron concentration control
- Safety limit monitoring (DNBR, PCT, LHGR)

### 7. Grid Interface (`components/grid_interface/`)
Power output management:
- Grid frequency regulation
- Power dispatch modeling

### 8. Master Clock (`components/master_clock/`)
Time synchronization across all components:
- Simulation time management
- Tick distribution

### 9. Telemetry (`components/telemetry_logging/`)
Data collection and logging:
- Real-time parameter monitoring
- Historical data storage

---

## Installation

### Prerequisites

- Python 3.8+
- OpenMC (for neutron transport)
- MPI implementation (for OpenMC parallel execution)

### Dependencies

Install required packages:

```bash
pip install numpy pandas pyyaml zmq pydantic openmc
```

For development/testing:

```bash
pip install pytest pytest-cov
```

### OpenMC Setup

OpenMC requires nuclear data libraries. Download and configure:

```bash
# Download OpenMC nuclear data
export OPENMC_CROSS_SECTION_DATA=/path/to/nuclear/data
```

---

## Configuration

Configuration files are located in `npp_simulation/config/`:

| File | Description |
|------|-------------|
| `reactor.yaml` | Reactor core parameters |
| `coolant.yaml` | Primary coolant properties |
| `steam_generator.yaml` | Steam generator settings |
| `turbine.yaml` | Turbine parameters |
| `condenser.yaml` | Condenser configuration |
| `control_system.yaml` | Control system parameters |
| `master_clock.yaml` | Time management |
| `telemetry.yaml` | Data logging settings |
| `runner.yaml` | Main runner configuration |

### Key Reactor Parameters

```yaml
parameters:
  P_rated_MW: 3400.0          # Rated thermal power (MW)
  phi_nominal: 1.0            # Nominal neutron flux
  flow_rate_kg_s: 17000.0     # Coolant mass flow rate
  Tin_coolant: 300.0          # Coolant inlet temperature (K)
  P_coolant_MPa: 15.5         # Coolant pressure (MPa)
  N_rods: 35000               # Number of fuel rods
```

---

## Running the Simulation

### Start the Runner

The main entry point is the runner which spawns all components:

```bash
cd npp_simulation
python -m runner.runner
```

### Individual Component Execution

Run components individually for testing:

```bash
# Run reactor core
python components/reactor_core/reactor.py

# Run primary coolant
python components/primary_coolant_loop/coolant.py
```

---

## Testing

Run the test suite:

```bash
cd npp_simulation
pytest tests/ -v
```

Available tests:
- `test_reactor.py` - Reactor core functionality
- `test_coolant.py` - Coolant loop physics
- `test_turbine.py` - Turbine modeling
- `test_master_clock.py` - Time synchronization
- `test_end_to_end.py` - Full system integration

---

## Project Structure

```
nuclear_sim/
├── README.md
├── npp_simulation/
│   ├── __init__.py
│   ├── lib/
│   │   ├── __init__.py
│   │   └── lib.py                    # Utility functions
│   ├── common/
│   │   ├── data_models.py           # Pydantic data models
│   │   ├── physics_formulas.py      # Physics calculations
│   │   ├── zmq_utils.py             # ZMQ utilities
│   │   ├── logger.py                 # Logging utilities
│   │   ├── time_service.py           # Time management
│   │   └── utils.py                  # Helper functions
│   ├── components/
│   │   ├── reactor_core/
│   │   │   ├── reactor.py            # Reactor with OpenMC
│   │   │   └── __init__.py
│   │   ├── primary_coolant_loop/
│   │   │   ├── coolant.py
│   │   │   └── __init__.py
│   │   ├── steam_generator/
│   │   │   ├── steam_generator.py
│   │   │   └── __init__.py
│   │   ├── turbine_generator/
│   │   │   ├── turbine.py
│   │   │   └── __init__.py
│   │   ├── condenser_cooling/
│   │   │   ├── condenser.py
│   │   │   └── __init__.py
│   │   ├── condenser_cooling_water_source/
│   │   ├── control_safety/
│   │   │   ├── control_system.py
│   │   │   └── __init__.py
│   │   ├── grid_interface/
│   │   ├── master_clock/
│   │   │   ├── master_clock.py
│   │   │   └── __init__.py
│   │   └── telemetry_logging/
│   ├── config/
│   │   ├── reactor.yaml              # Main reactor config
│   │   ├── coolant.yaml
 ├── steam_generator.yaml│   │  
│   │   ├── turbine.yaml
│   │   ├── condenser.yaml
│   │   ├── grid.yaml
│   │   ├── control_system.yaml
│   │   ├── master_clock.yaml
│   │   ├── telemetry.yaml
│   │   ├── runner.yaml
│   │   └── condenser_cooling_water_source.yaml
│   ├── runner/
│   │   ├── runner.py                 # Main orchestration
│   │   ├── test.py
│   │   ├── test_2.py
│   │   └── test_3.py
│   ├── tests/
│   │   ├── test.py
│   │   ├── test_2.py
│   │   ├── test_3.py
│   │   ├── test_reactor.py
│   │   ├── test_coolant.py
│   │   ├── test_turbine.py
│   │   ├── test_master_clock.py
│   │   ├── test_end_to_end.py
│   │   ├── test_read.py
│   │   ├── tallies.xml
│   │   ├── geometry.xml
│   │   ├── materials.xml
│   │   ├── settings.xml
│   │   └── plots.xml
│   ├── prediction_engine/
│   │   ├── model_training.py        # ML model training
│   │   ├── model_inference.py       # Real-time inference
│   │   ├── feature_engineering.py   # Feature extraction
│   │   └── __init__.py
│   ├── dashboard/
│   │   ├── app.py
│   │   ├── charts.py
│   │   └── __init__.py
│   ├── requirements.txt
│   ├── docker-compose.yml
│   ├── .env
│   ├── tree.txt
│   ├── output_test_reactor.txt
│   └── tree.txt
├── data/
│   ├── nuclear_data_1/              # US nuclear statistics
│   ├── nuclear_data_3/
│   ├── nuclear_data_5/              # Power consumption data
│   ├── nuclear_data_6/              # Generation data
│   └── other/
│       ├── French nuclear reactors availability
│       ├── Global power plants
│       ├── Chernobyl data
│       └── World nuclear reactors
└── R24DE201/
    └── test_data_log_file-50.pkl
```

---

## Key Features

### 1. Monte Carlo Neutron Transport
- Full OpenMC integration for accurate criticality calculations
- K-eff and reactivity computations
- Energy spectrum analysis
- Flux and heating tallies

### 2. Point Kinetics Model
- Time-dependent neutron flux calculation
- Delayed neutron precursor groups (6 groups)
- Reactivity feedback

### 3. Thermal-Hydraulics
- Multi-stage heat transfer (fuel → cladding → coolant)
- DNBR (Departure from Nucleate Boiling Ratio) monitoring
- PCT (Peak Cladding Temperature) tracking

### 4. Control Rod Modeling
- 6 control rod banks (A-F) with individual worth tables
- Insertion depth tracking
- Speed-based insertion/withdrawal

### 5. Real-Time Simulation
- ZMQ-based distributed architecture
- Sub-second time step simulation
- Heartbeat monitoring for all components

### 6. Comprehensive Telemetry
- Real-time data logging
- Pickle-formatted data export
- Statepoint file generation (HDF5)

---

## Data

The project includes reference data in `data/`:

- **US Nuclear Generating Statistics** (1971-2021)
- **World Nuclear Energy Generation**
- **French Nuclear Reactors Availability**
- **Global Power Plants Database**
- **Uranium Production Data**
- **Chernobyl Incident Data**

---

## Technology Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.8+ |
| Neutron Transport | OpenMC |
| Message Queue | ZeroMQ |
| Data Validation | Pydantic |
| Configuration | YAML |
| Testing | pytest |
| Numerical Computing | NumPy |
| Data Analysis | Pandas |
| Visualization | Matplotlib/Plotly |

---

## Configuration Example: Control Rod Banks

```yaml
rod_insertion_depth:
  banks:
    - name: "Bank A"
      rods: 50
      depth: 0.0      # 0% = fully withdrawn
      length: 4000.0  # mm
      worth_table:
        x: [0.0, 0.2, 0.5, 0.8, 1.0]
        y: [0, -200, -1200, -2500, -3000]  # pcm
      speed: 0.05     # m/s
```

---

## Notes

- The `requirements.txt` may be empty - install dependencies manually as listed above
- OpenMC requires proper nuclear data libraries for full functionality
- MPI is needed for parallel OpenMC execution
- The dashboard components are placeholders for future UI development

---

## License

This project is for educational and research purposes.
