import pandas as pd
import pathlib
import random

class DataGenerator:
    def __init__(self) -> None:
        self.data = {
                        "Time Since Last Maintenance(in months)": [],
                        "Operational Hours": [],
                        "Wear Level": [],
                        "Error Rate": []
                    }
    
    def generate_samples(self, num_samples, months_since_last_maintenance, max_operational_hours, max_wear_level, max_error_rate, savefilename):
        for _ in range(num_samples):
            time_since_last_maintenance_months = random.randint(1, months_since_last_maintenance)
            operational_hours = random.randint(0, max_operational_hours)
            wear_level = random.randint(0, max_wear_level)
            error_rate = random.randint(0, max_error_rate)

            self.data["Time Since Last Maintenance(in months)"].append(time_since_last_maintenance_months)
            self.data["Operational Hours"].append(operational_hours)
            self.data["Wear Level"].append(wear_level)
            self.data["Error Rate"].append(error_rate)

        df = pd.DataFrame(self.data)
        df.to_csv(savefilename, index=False)
        print(f"Dataset created and saved as {savefilename}.")
