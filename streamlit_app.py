import streamlit as st
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

model= joblib.load('K-Nearest Neighbors.pkl',)

with open