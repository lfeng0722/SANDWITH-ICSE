# SANDWITH-ICSE

## 📁 Repository Structure

- `.idea/`  
  *Project-specific settings for JetBrains IDEs.*

- `MARL/`  
  *Contains MARL weight*

- `__pycache__/`  
  *Directory for Python bytecode cache files.*

- `map/`  
  *Includes map files used in simulations.*

- `ART_fuzzer.py`  
  *Script for the Adaptive Random Testing fuzzer.*

- `ICSE_2026_Supplementary.pdf`  
  *Supplementary material for the ICSE 2026 submission.*

- `MADDPG.py`  
  *Implementation of the Multi-Agent Deep Deterministic Policy Gradient algorithm.*

- `README.md`  
  *Provides an overview and setup instructions for the project.*

- `agent.py`  
  *Defines agent behaviors and interactions.*

- `buffer.py`  
  *Manages experience replay buffers.*

- `carla_controller.py`  
  *Controls the CARLA simulator environment.*

- `demographic information.csv`  
  *CSV file containing demographic data.*

- `fuzzer_set_mapper.py`  
  *Maps fuzzer configurations to specific settings.*

- `listener.py`  
  *Script for listening to simulation events.*

- `main.py`  
  *Main entry point for executing the system.*

- `manual_control.py`  
  *Script for manual control within the simulation.*

- `math_tool.py`  
  *Provides mathematical utilities and functions.*

- `networks.py`  
  *Defines neural network architectures.*

- `offline_searcher.py`  
  *Performs offline search operations.*

- `requirements.txt`  
  *Lists Python dependencies required for the project.*

- `sim_env.py`  
  *Sets up the simulation environment.*

- `tick_manager.py`  
  *Manages simulation ticks and timing.*

- `utility.py`  
  *Contains utility functions used across the project.*

- `Raw_data.xlsx`  
  *Contains raw result of our experiments*

- `Statistical_Testing.xlsx`  
  *Contains Statistical result of our experiments*

- 
## Installation

### Install Packages  
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 1. Install CARLA  
First, install **CARLA 0.9.13** from the official website: [CARLA](https://carla.org/).

### 2. Install Apollo  
Next, install **Apollo 8.0** from the official repository: [Apollo](https://github.com/ApolloAuto/apollo).

### 3. Install CARLA-Apollo Bridge  
Install the **CARLA-Apollo Bridge** from: [CARLA Apollo Bridge](https://github.com/MaisJamal/carla_apollo_bridge).

### 4. File Setup  
- Copy `manual_control.py` to `/carla_apollo_bridge/example`.  
- Copy `listener.py` to the Apollo root folder.  
- Copy the maps in the `map` folder to `Apollo/modules/map/data`.

## Running the System  

1. Start **CARLA**:  
   ```bash
   ./CarlaUE4.sh -RenderOffScreen
   ```
2. Run **tick_manager**:  
   ```bash
   python tick_manager
   ```
3. Run the CARLA configuration script in the **bridger docker**. Change the map name if you are using a different map:  
   ```bash
   python carla-python-0.9.13/util/config.py -m Town04 --host 172.17.0.1
   ```
4. Start the **manual control script**:  
   ```bash
   python examples/manual_control.py
   ```
5. Launch **Apollo** by following the instructions on the [CARLA Apollo Bridge repository](https://github.com/MaisJamal/carla_apollo_bridge).  
6. Run the **CARLA Cyber Bridge** in another **bridge docker**:  
   ```bash
   python carla_cyber_bridge/run_bridge.py
   ```
7. Start the **listener script** in the Apollo docker:  
   ```bash
   python listener.py
   ```
   - If you encounter issues running `listener.py`, refer to the official Apollo documentation: [Apollo Cyber Python README](https://github.com/ApolloAuto/apollo/blob/master/cyber/python/README.md).

8. Run the **main script**:  
   ```bash
   python main.py
   