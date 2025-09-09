make:
	nvcc -o cuda_program.o cuda/$(file).cu
	./cuda_program.o

git:
	git add .
	git commit -m "$(msg)"
	git push