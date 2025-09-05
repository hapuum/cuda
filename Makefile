make:
	nvcc -o cuda_program.o $(file).cu

git:
	git add .
	git commit -m "$(msg)"
	git push