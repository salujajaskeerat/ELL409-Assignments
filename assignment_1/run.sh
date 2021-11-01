args=""
for ITEM in "$@"
do
    args="$args $ITEM" 
done

python main.py $args