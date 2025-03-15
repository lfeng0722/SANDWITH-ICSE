# SANDWITH-ICSE

## Installation

### 1. Install CARLA  
First, install **CARLA 0.9.15** from the official website: [CARLA](https://carla.org/).

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
