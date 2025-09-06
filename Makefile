make:
	nvcc -o cuda_program.o $(file).cu
	./cuda_program.o

git:
	git add .
	git commit -m "$(msg)"
	git push