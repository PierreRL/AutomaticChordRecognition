set +e
#python src/run.py --input_dir=./data/processed --output_dir=./results/cnn --model=cnn --exp_name=cnn_k5_l1_c1 --cnn_kernel_size=5 --num_layers=1 --cnn_channels=1 
python src/run.py --input_dir=./data/processed --output_dir=./results/cnn --model=cnn --exp_name=cnn_k9_l5_c10 --cnn_kernel_size=9 --num_layers=5 --cnn_channels=10 --seed=1
#python src/run.py --input_dir=./data/processed --output_dir=./results/cnn --model=cnn --exp_name=cnn_k5_l3_c5 --cnn_kernel_size=5 --num_layers=3 --cnn_channels=5 
#python src/run.py --input_dir=./data/processed --output_dir=./results/cnn --model=cnn --exp_name=cnn_k15_l7_c20 --cnn_kernel_size=15 --num_layers=7 --cnn_channels=20 
#python src/run.py --input_dir=./data/processed --output_dir=./results/cnn --model=cnn --exp_name=cnn_k11_l15_c5 --cnn_kernel_size=11 --num_layers=15 --cnn_channels=5