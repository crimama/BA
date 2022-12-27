python TE.py -Exp TE0 -model_name PiModel -unlabel_ratio 0 -super_only True
python TE.py -Exp TE1 -model_name PiModel -unlabel_ratio 0 
python TE.py -Exp TE2 -model_name PiModel -unlabel_ratio 0.3 
python TE.py -Exp TE3 -model_name PiModel -unlabel_ratio 0.6 
python TE.py -Exp TE4 -model_name PiModel -unlabel_ratio 0.9 

python TE.py -Exp TE5 -model_name resnet18 -unlabel_ratio 0 -super_only True
python TE.py -Exp TE6 -model_name resnet18 -unlabel_ratio 0 
python TE.py -Exp TE7 -model_name resnet18 -unlabel_ratio 0.3 
python TE.py -Exp TE8 -model_name resnet18 -unlabel_ratio 0.6 
python TE.py -Exp TE9 -model_name resnet18 -unlabel_ratio 0.9 