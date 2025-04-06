# npc_take_home
Hi, I am Pancy Singla and this is my script for the Strategy I have used and the configuration file for the same.

# Hummingbot Python Script

## Setup & Usage

1. Copy these files to your Hummingbot directory:
   - Copy your Python script to the scripts folder: `/hummingbot/scripts/`
   - Copy your config file to the config folder: `/hummingbot/conf/`

2. Start Hummingbot:
   ```
   cd /path/to/hummingbot
   ./start
   ```

3. Inside the Hummingbot terminal, create a script configuration:
   ```
   create --script-config scriptname
   ```
   
4. Fill in the parameters when prompted. Your config file already contains the parameters you've used.

5. Start running the script with your configuration:
   ```
   start --script scriptname --conf your_config_filename
   ```

The script will execute your trading strategy using the parameters from your config file.
