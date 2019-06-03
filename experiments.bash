for i in {1..8}
do
   python mnist.py --dilation $i --dataset mnist
done

for i in {1..8}
do
   python mnist.py --dilation $i --dataset mnistfashion
done
