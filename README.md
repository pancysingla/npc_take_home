# npc_take_home
Hi, I am Pancy Singla and this is my script for the Strategy I have used and the configuration file for the same.

# Hummingbot Python Script

## Setup & Usage

1. Copy these files to your Hummingbot directory:
   - Copy your Python script to the scripts folder: `/hummingbot/scripts/`
   - Copy your config file to the config folder: `/hummingbot/conf/scripts`

2. Start Hummingbot:
   ```
   cd /path/to/hummingbot
   ./start
   ```

3. Inside the Hummingbot terminal, create a script configuration:
   ```
   create --script-config finn_pmm.py
   ```
   
4. Fill in the parameters when prompted. Your config file already contains the parameters you've used.

5. Start running the script with your configuration:
   ```
   start --script fin_pmm.py --conf conf_fin_pmm_1.yml
   ```

The script will execute your trading strategy using the parameters from your config file.
