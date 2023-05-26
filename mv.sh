for i in L*
do
    echo $i
    mv $i/* ./
    rm -r $i
done