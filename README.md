## Aviation System of Systems: Implications of Inter-Aircraft Communcation 

This repository contains a Python-based simulation developed as part of a Master’s thesis at **Linköping University, Sweden**.  
The project investigates **inter-aircraft communication and decentralized en-route weather avoidance** from a **System-of-Systems (SoS)** perspective.

The model focuses on how short, peer-level information exchanges between aircraft influence operational efficiency and safety when encountering unexpected and localized atmospheric disturbances.

## Overview
Aircraft are represented as autonomous agents moving through a corridor-based airspace network.  
Each agent can sense disturbances, exchange minimal operational advisories with nearby aircraft, and adapt its route locally.  
The simulation is intentionally lightweight and transparent, emphasizing behavioral mechanisms and emergent effects rather than avionics-level detail.

Key aspects include:
- Corridor-based routing and local rerouting
- Decentralized peer-to-peer (P2P) information sharing
- VFR-based aircraft movement and communication implemented
- Three tested scenarios (non-exsistant, instant, and delayed P2P communication)
- Weather-induced disturbances (e.g., cumulonimbus cells)
- Performance evaluation via delay and avoidance-related KPIs
- Batch execution and CSV-based result logging for reproducibility

## File Structure

- `aircraft_simulation_croatia_extended.py`  
  Main simulation script containing model setup, agent logic, and KPI logging.


- `LICENSE`  
  MIT License.

- `CITATION.cff`
  Citation metadata for academic reuse.


## Requirements and Code Running 

- Python 3.9 or newer
- This code was directly run in IDLE
- Common scientific Python libraries (e.g., `numpy`, `networkx`, `pandas`, `matplotlib`)
- Line 9 toggle BATCH_MODE between 0 and 1 to run one run or in batches (set BATCH_SEEDS_STR range for number of runs in batch)
- Line 20 toggle NO_PLOT to show or hide animation during running
