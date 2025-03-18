#!/bin/bash

ssh -L 6006:localhost:47761 "$1"
