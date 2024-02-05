from abc import ABC, abstractmethod

def get_factor(unit:int = None, byte_unit:str='kB'):
    factor = 1
    if unit: 
        factor *= unit
    
    if byte_unit:
        if byte_unit.upper() == 'MB': 
            factor *= 1024
        elif byte_unit.upper() == 'GB': 
            factor *= 1024 * 1024

    return factor

class DBConnector(ABC):
    @abstractmethod
    def connect_db(self):
        pass

    @abstractmethod
    def close_db(self):
        pass

    @abstractmethod
    def fetch_results(self, sql, json=True):
        pass

    @abstractmethod
    def execute(self, sql):
        pass
