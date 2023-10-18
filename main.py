import gradio as gr
from scipy.fftpack import fft
import numpy as np
import os
from math import log2, pow
from datasets import load_dataset
from utils import *

asr = AutomaticSpeechRecognition()

simple_ui(asr)
