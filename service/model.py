from model import PlaneLSTMModule, device
from config import num_hiddens, PARAMS_PATH
from data_loader import NUM_FEATURES


def get_model():
    model = PlaneLSTMModule(num_hiddens, NUM_FEATURES)
    model.load_parameters(PARAMS_PATH, ctx=device)
    return model
