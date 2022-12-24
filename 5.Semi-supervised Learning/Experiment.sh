python main.py -Exp 0 -model 'resnet18' -unlabel_ratio 0
python main.py -Exp 1 -model 'resnet18' -unlabel_ratio 0.3
python main.py -Exp 2 -model 'resnet18' -unlabel_ratio 0.6
python main.py -Exp 3 -model 'resnet18' -unlabel_ratio 0.9

python main.py -Exp 0 -model 'ssl_resnet50' -unlabel_ratio 0
python main.py -Exp 1 -model 'ssl_resnet50' -unlabel_ratio 0.3
python main.py -Exp 2 -model 'ssl_resnet50' -unlabel_ratio 0.6
python main.py -Exp 3 -model 'ssl_resnet50' -unlabel_ratio 0.9