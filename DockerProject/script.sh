#!/bin/bash


exec python -u FlaskSocketIO_GUI.py &
exec python -u FlaskSocketIO_backend.py

wait