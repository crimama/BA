python Pi.py -Exp Pi0 -model_name PiModel -unlabel_ratio 0 -super_only True
python Pi.py -Exp Pi1 -model_name PiModel -unlabel_ratio 0 
python Pi.py -Exp Pi2 -model_name PiModel -unlabel_ratio 0.3 
python Pi.py -Exp Pi3 -model_name PiModel -unlabel_ratio 0.6 
python Pi.py -Exp Pi4 -model_name PiModel -unlabel_ratio 0.9 

python Pi.py -Exp Pi5 -model_name resnet18 -unlabel_ratio 0 -super_only True
python Pi.py -Exp Pi6 -model_name resnet18 -unlabel_ratio 0 
python Pi.py -Exp Pi7 -model_name resnet18 -unlabel_ratio 0.3 
python Pi.py -Exp Pi8 -model_name resnet18 -unlabel_ratio 0.6 
python Pi.py -Exp Pi9 -model_name resnet18 -unlabel_ratio 0.9 