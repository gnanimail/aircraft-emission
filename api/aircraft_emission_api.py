import argparse
import joblib
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error
import streamlit as st

st.title("Prediction of NOx Emission Index During Climb Phase of the Flight")

# defining the function which will make the prediction using the data which the user inputs
@st.cache()
def prediction(model, Nox_Dp_Foo_Characteristic, Ambient_Baro_Max, Fuel_Flow_CO, Fuel_LTO_Cycle, NOx_LTO_Total_mass,
                   B_P_Ratio, Rated_Thrust, Ambient_Temp_Min, Ambient_Baro_Min, Fuel_Arom_Min, Fuel_Arom_Max, SN_C_O,
                   Fuel_HC_Ratio_Min, Pressure_Ratio, HumidityMin):

    prediction_api = model.predict(
            np.array([[Nox_Dp_Foo_Characteristic, Ambient_Baro_Max, Fuel_Flow_CO, Fuel_LTO_Cycle, NOx_LTO_Total_mass,
                       B_P_Ratio, Rated_Thrust, Ambient_Temp_Min, Ambient_Baro_Min, Fuel_Arom_Min, Fuel_Arom_Max,
                       SN_C_O, Fuel_HC_Ratio_Min, Pressure_Ratio, HumidityMin]]))

    output = round(prediction_api[0], 2)
    return output


def ae_api(model_path):

    # loading the trained model
    # pickle_in = open('ae_ml.pkl', 'rb')
    # model = pickle.load(pickle_in)
    model = joblib.load(model_path)

    Nox_Dp_Foo_Characteristic = st.number_input("Nox_Dp / Foo_Characteristic - Oxides of nitrogen characteristic Dp/Foo value. Input Range (20, 110)")
    Ambient_Baro_Max = st.number_input("Ambient_Baro_Max - Maximum ambient pressure (kPA).   Input range (90-110)")
    Fuel_Flow_CO = st.number_input("Fuel_Flow_C / O - Fuel flow (kg/sec) at climb out condition.   Input range (0-5)")
    Fuel_LTO_Cycle = st.number_input("Fuel_LTO_Cycle - kg of fuel used during the Landing Take-Off cycle.   Input range (300-1600)")
    NOx_LTO_Total_mass = st.number_input("NOx_LTO_Total_mass - The total mass of oxides of nitrogen emitted during the LTO cycle (sum of time in mode x fuel flow x EI at each of the four power settings).  Input range (900-35000)")
    B_P_Ratio = st.number_input("B/P_Ratio - Bypass ratio.  Input range (0-15)")
    Rated_Thrust = st.number_input("Rated_Thrust - Engine maximum rated thrust, in kilonewtons.  Input range (20-550)")
    Ambient_Temp_Min = st.number_input("Ambient_Temp_Min - Minimum temperature (K).  Input range (200-300)")
    Ambient_Baro_Min = st.number_input("Ambient_Baro_Min - Minimum ambient pressure (kPA).  Input range (90-110)")
    Fuel_Arom_Min = st.number_input("Fuel_Arom_Min(%) - Minimum percentage of aromatic hydrocarbons in the fuel.  Input range (10-25)")
    Fuel_Arom_Max = st.number_input("Fuel_Arom_Max(%) - Maximum percentage of aromatic hydrocarbons in the fuel.  Input range (10-25)")
    SN_C_O = st.number_input("SN_C/O - Smoke number at climb out condition.  Input range (0-50)")
    Fuel_HC_Ratio_Min = st.number_input("Fuel_H/C_Ratio_Min - Maximum ambient humidity in kg water per kg dry air.  Input range (0-200)")
    Pressure_Ratio = st.number_input("Pressure_Ratio - Engine pressure ratio.  Input range (0-50)")
    HumidityMin = st.number_input("HumidityMin - Minimum ambient humidity in kg water per kg dry air.  Input range (0-0.05)")
    result = ""

    # when 'Predict' is clicked, make the prediction and store it
    if st.button("Predict"):
        result = prediction(model, Nox_Dp_Foo_Characteristic, Ambient_Baro_Max, Fuel_Flow_CO, Fuel_LTO_Cycle,
                                NOx_LTO_Total_mass, B_P_Ratio, Rated_Thrust, Ambient_Temp_Min, Ambient_Baro_Min,
                                Fuel_Arom_Min, Fuel_Arom_Max, SN_C_O, Fuel_HC_Ratio_Min, Pressure_Ratio, HumidityMin)

        if result < 0:
            st.success('Predictions could not be made '.format(result))
        else:
            st.success('NOx Emission index during Climb phase {}'.format(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    args = parser.parse_args()
    ae_api(args.model)
