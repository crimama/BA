python TE.py -Exp TE0 -model_name PiModel -unlabel 5000 -unlabel 0 
python TE.py -Exp TE1 -model_name PiModel -unlabel 5000 -unlabel 1000 
python TE.py -Exp TE2 -model_name PiModel -unlabel 5000 -unlabel 10000 
python TE.py -Exp TE3 -model_name PiModel -unlabel 5000 -unlabel 25000 
python TE.py -Exp TE4 -model_name PiModel -unlabel 5000 -unlabel 45000 

python TE.py -Exp TE5 -model_name resnet18 -unlabel 5000 -unlabel 0 
python TE.py -Exp TE6 -model_name resnet18 -unlabel 5000 -unlabel 1000 
python TE.py -Exp TE7 -model_name resnet18 -unlabel 5000 -unlabel 10000
python TE.py -Exp TE8 -model_name resnet18 -unlabel 5000 -unlabel 25000
python TE.py -Exp TE9 -model_name resnet18 -unlabel 5000 -unlabel 45000