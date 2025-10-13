make:
	nvcc -o cuda_program.o json_parser.cu
	./cuda_program.o

debug:
	nvcc -G -g -O0 -lineinfo -o debug_program.o json_parser.cu

git:
	git add .
	git commit -m "$(msg)"
	git push