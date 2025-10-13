make:
	nvcc -o cuda_program.o json_parser.cu
	./cuda_program.o

git:
	git add .
	git commit -m "$(msg)"
	git push