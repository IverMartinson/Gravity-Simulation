clear
echo "[BUILD] Building"
if gcc -g source.cu -o build -lSDL2; then
	echo "[BUILD] Running Binary"
	./build
fi
