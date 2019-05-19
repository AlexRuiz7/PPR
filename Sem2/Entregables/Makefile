# File Makefile

# Compiler flags
FLAGS= -O3

# Compiler
CC = mpiCC

# Macros
SOURCES = 1.cc 2.cc 3.cc 4.cc

.PHONY: all clean

# Compilation
all: 1 2 3 4

1:
	$(CC) 1.cc -o 1 $(FLAGS)

2:
	$(CC) 2.cc -o 2 $(FLAGS)

3:
	$(CC) 3.cc -o 3 $(FLAGS)

4:
	$(CC) 4.cc -o 4 $(FLAGS)

# Utilities
# Remove executable files
clean:
	@echo "Removing object files..."
	rm -f 1 2 3 4

# Compress
tgz: clean
	@echo "Compressing files..."
	tar -zcvf Sem2.tgz *
