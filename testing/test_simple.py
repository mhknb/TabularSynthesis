import streamlit as st
import pandas as pd
import numpy as np
import time

st.title("Simple Test App")

st.write("This is a basic test app.")

# Create a simple data display
data = {
    'A': [1, 2, 3, 4, 5],
    'B': [10, 20, 30, 40, 50]
}

df = pd.DataFrame(data)
st.dataframe(df)

# Add a button that does something
if st.button("Click me"):
    st.success("Button was clicked!")
    
    # Show a progress bar
    progress = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress.progress(i + 1)
    
    st.balloons()