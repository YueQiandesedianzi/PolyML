import sys
sys.path.insert(0, 'E:/0-DHU/CC/backend')
from main import app
print('FastAPI app created OK')
from ml.models import MODEL_DEFINITIONS
print(f'Models defined: {list(MODEL_DEFINITIONS.keys())}')
from ml.descriptors import RDKitDescriptorCalculator
calc = RDKitDescriptorCalculator()
print(f'RDKit descriptors: {len(calc.descriptor_names)}')
from ml.van_krevelen import VanKrevelenEngine
vk = VanKrevelenEngine()
print(f'VK groups: {len(vk.groups)}')
print('All imports OK')
