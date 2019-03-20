#include <iostream>
#include "mpi.h"

int main(int argc, char *argv[]) {
	MPI_Init(&argc, &argv);
	int rank, communicatorSize;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &communicatorSize);

	if(rank == 0) rank0(communicatorSize);
	else if (rank==1) errorHandlerRank(1);
	else ranki(rank);

	MPI_Finalize();
	return 0;
}


void rank0(int communicatorSize)
{

}

void ranki(int rank)
{

}
void errorHandlerRank(int rank)
{

}