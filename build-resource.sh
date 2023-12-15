export RESOURCE=$PWD/resource.bfs

# clean resource
rm $RESOURCE

echo NN
cd resource
./build-resource.sh $RESOURCE
cd ..

echo CONTENTS
bfs $RESOURCE blobList
