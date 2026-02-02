# Multi-Agent Simulator for Territorial Input Output

## Project Description
MASTIO is a simulation platform designed to model agent-based interactions in a market environment. It simulates buyers and sellers exchanging products, with agents learning and adapting their strategies over time. The platform includes features for visualizing market dynamics and analyzing agent behavior.

![Simulation](https://exemple.com/monimage.png)

## Installation

### Prerequisites
- Python 3.x
- pip (Python package installer)

### Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/AAMAS746/aamas2026-746.git
   cd MASTIO 
   ```
2. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv .venv
   ```
3. **Activate the virtual environment:**
   - On macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```
   - On Windows:
     ```bash
     .venv\Scripts\activate
     ```
4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running a Simulation
Simulations are configured using YAML files. An example configuration file (`conf_example.yaml`) is provided.

To run a simulation:
```bash
python MASTIO/run.py conf_example.yaml
```

#### `data/products.csv` 
You need to provide a product file that lists all available products and their characteristics. An example is provided in data/products_example.csv.

- **name**: Name of the product  
- **sellers_rate**: The rate of agents discarding this product  
- **buyers_rate**: The rate of agents needing this product  
- **market_price**: The price at which this product can be bought on the main market  
- **km_cost**: The transportation cost per km for this product
- **scarcity**: The availability of the product. A scarcity of 2 means demand is twice as high as supply  
- **price_to_dispose**: The price a seller must pay if they cannot sell the product


### Output
Simulation results, including plots and regret analysis data, are saved in the directory specified by `processed_path` in the configuration file.

## Project Structure
The core simulation logic resides in the `MASTIO/` directory:

- `agent.py`: Defines the `Agent` class, representing market participants (buyers and sellers) and their learning mechanisms.
- `area.py`: Defines the `SquareArea` class, used for defining the geographical space of the simulation.
- `auction.py`: Implements the `Auction` mechanism for matching buyers and sellers for a specific product.
- `model.py`: The main simulation model, orchestrating agent creation, simulation steps, and data collection.
- `regret_analysis.py`: Provides tools for analyzing agent regret, including counterfactual simulations.
- `run.py`: The entry point for running the simulation from the command line.
- `utils.py`: Utility functions, including distance calculations and normalization.
- `visualization.py`: Functions for generating various plots and animations of simulation results.

## Contributing
Contributions are welcome! Please refer to the project's contribution guidelines (if available) for more details.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
