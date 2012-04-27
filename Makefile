CC=clang
CFLAGS=-O2 -Wall
LDFLAGS=-lm -lJudy

all: snet

clean: 
	rm -rf *.o snet *.sum *.log *.est *.sol
