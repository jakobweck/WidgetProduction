#include <iostream>
#include "mpi.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <array>
#include <vector>
#include <math.h>

using namespace std;
struct Data
{
	int modelNum;
	int purchaseDate;
	char customerType;
	float amount;
};

void defineDataStruct(MPI_Datatype* structType){
	int blklen[4];
	MPI_Aint displ[4];
	MPI_Datatype types[4];

	blklen[0]=1;
	blklen[1]=1;
	blklen[2]=1;
	blklen[3]=1;

	displ[0] = 0;
	Data sample;
	MPI_Aint base;
	MPI_Get_address(&sample.modelNum, &base);
	MPI_Aint oneField;
	MPI_Get_address(&sample.purchaseDate, &oneField);
	displ[1] = oneField-base;
	MPI_Get_address(&sample.customerType, &oneField);
	displ[2] = oneField-base;
	MPI_Get_address(&sample.amount, &oneField);
	displ[3] = oneField-base;

	types[0] = MPI_INT;
	types[1] = MPI_INT;
	types[2] = MPI_DOUBLE;
	types[3] = MPI_FLOAT;

	MPI_Type_create_struct(4, blklen, displ, types, structType);
	MPI_Type_commit(structType);
}

string floatToString(float num,int precision){
	std::stringstream ss;
	ss << std::fixed << std::setprecision(precision) << num;
	std::string res = ss.str();
	return res;
}

void printResult(float* total, float* forYear, float* forType, int numModels, int yearToReport, char typeToReport)
{
	ostringstream os;
	MPI_Barrier(MPI_COMM_WORLD);
	os << "Model Num ---- Total Sales ---- Sales for " << yearToReport << " ---- Sales for type " << typeToReport << endl;
	for(int i =0;i<numModels;i++)
	{
		string thisTotal = "$" + floatToString(total[i], 2);
		string yearTotal = "$" + floatToString(forYear[i], 2);
		string typeTotal = "$" + floatToString(forType[i], 2);
		os <<setw(9)<< i;
		os << " |||| ";
		os <<setw(11)<< thisTotal;
		os << " |||| ";
		os <<setw(14)<< yearTotal;
		os << " |||| ";
		os <<setw(16)<< typeTotal;
		os << endl;
	}
	cout << os.str();
	return;
}

string dataString(Data row)
{
	return to_string(row.modelNum) + " " + to_string(row.purchaseDate) + " " + row.customerType + " " + to_string(row.amount);
}

void processRows(int rank, MPI_Comm comm, Data* rows, int sliceSize, int numModels, MPI_Datatype dataStruct, int yearToReport, char typeToReport){
	float* modelNumTotalSales = new float[numModels];
	float* modelNumSalesSpecYear = new float[numModels];
	float* modelNumSalesSpecType = new float[numModels];
	for(int i=0;i<numModels;i++)
	{
		modelNumTotalSales[i] = 0;
		modelNumSalesSpecYear[i] = 0;
		modelNumSalesSpecType[i] = 0;
	}
	for(int i=0;i<sliceSize;i++)
	{
		//error message dispatching to error reporter process
		Data currRow = rows[i];
		bool errors[6] = {false, false, false, false, false, false};
		errors[0] = (currRow.modelNum <0 || currRow.modelNum > numModels);
		errors[1] = ((to_string(currRow.purchaseDate)).length() != 6);
		errors[2] = (currRow.purchaseDate%100 < 1 || currRow.purchaseDate%100 > 12);
		errors[3] = (currRow.purchaseDate<199701 || currRow.purchaseDate>201812);
		errors[4] = (currRow.customerType != 'G' && currRow.customerType != 'I' && currRow.customerType != 'R');
		errors[5] = (currRow.amount <= 0.0);
		bool anyErrors = false;
		for(int j=0;j<6;j++)
		{
			if(errors[j])
			{
			//	cout << "Rank " << rank << " error" << endl;
				anyErrors = true;
				MPI_Request sendRequest;
				MPI_Status status;
				MPI_Isend(&currRow, 1, dataStruct, 1, j, MPI_COMM_WORLD, &sendRequest);
				MPI_Wait(&sendRequest, &status);
			}
		}
		if(anyErrors)
		{
			continue;
		}
		//data processing

		int currModelNum = currRow.modelNum;
		int currSale = currRow.amount;
		int currYear = (currRow.purchaseDate)/100;
		modelNumTotalSales[currModelNum] += currSale;
		if(currRow.customerType==typeToReport)
		{
			modelNumSalesSpecType[currModelNum] += currSale;
		}
		if(currYear == yearToReport)
		{
			modelNumSalesSpecYear[currModelNum] += currSale;
		}
	}
	//reduce
	if(rank==0)
	{
		MPI_Request sendRequest;
		MPI_Status status;
		MPI_Isend(nullptr, 0, dataStruct, 1, 6, MPI_COMM_WORLD, &sendRequest);
		MPI_Wait(&sendRequest, &status);
		float* modelNumTotalSalesR = new float[numModels];
		float* modelNumSalesSpecYearR = new float[numModels];
		float* modelNumSalesSpecTypeR = new float[numModels];
		MPI_Reduce(modelNumTotalSales, modelNumTotalSalesR, numModels, MPI_FLOAT, MPI_SUM, 0, comm);
		MPI_Reduce(modelNumSalesSpecYear, modelNumSalesSpecYearR, numModels, MPI_FLOAT, MPI_SUM, 0, comm);
		MPI_Reduce(modelNumSalesSpecType, modelNumSalesSpecTypeR, numModels, MPI_FLOAT, MPI_SUM, 0, comm);
		printResult(modelNumTotalSalesR, modelNumSalesSpecYearR, modelNumSalesSpecTypeR, numModels, yearToReport, typeToReport);

	}
	else
	{
		MPI_Reduce(modelNumTotalSales, nullptr, numModels, MPI_FLOAT, MPI_SUM, 0, comm);
		MPI_Reduce(modelNumSalesSpecYear, nullptr, numModels, MPI_FLOAT, MPI_SUM, 0, comm);
		MPI_Reduce(modelNumSalesSpecType, nullptr, numModels, MPI_FLOAT, MPI_SUM, 0, comm);
	}
}

void rank0(int communicatorSize, string filename, int yearToReport, char typeToReport)
{
	MPI_Comm MPI_COMM_WORKERS;
	MPI_Comm_split(MPI_COMM_WORLD, 0, 0, &MPI_COMM_WORKERS);

	MPI_Datatype dataStruct;
	defineDataStruct(&dataStruct);

	ifstream dataFile;
	dataFile.open(filename);

	if(dataFile.fail())
	{
		cout << "Failed to read file. Terminating processes." << "\n";
		MPI_Request failReq[2];
		int failMsg[2] = {-1, -1};
		MPI_Ibcast(failMsg, 2, MPI_INT, 0, MPI_COMM_WORKERS, &failReq[0]);
		return;
	}
	string firstLine;
	vector<string> remainingLines;
	int numRecords = 0;
	int numModels = 0;
	vector<Data> allRowsVec;
	if(dataFile.is_open())
	{
		int totalCount = 0;
		int count = 0;
		string word;
		Data* entry;
		while (dataFile >> word)
		{
			if(totalCount ==0)
			{
				numRecords = stoi(word);
				totalCount++;
			}
			else if (totalCount == 1)
			{
				numModels = stoi(word);
				totalCount++;
			}
			else if (totalCount >1)
			{
				switch(count)
				{
					case 0:
						entry = new Data();
						entry->modelNum = stoi(word);
						break;
					case 1:
						entry->purchaseDate = stoi(word);
						break;
					case 2:
						entry->customerType = word[0];
						break;
					case 3:
						entry->amount = stof(word);
						allRowsVec.push_back(*entry);
						break;
				}
				count = (count+1)%4;
				totalCount++;
			}

		}
		dataFile.close();
	}
	Data* allRows = allRowsVec.data();
	int sliceSize = (numRecords/(communicatorSize-1));
	//sliceSize is floored if the num of records can't be divided evenly
	//this many rows are left over and need to be handled somewhere
	int rowsLeftOver = (numRecords-(sliceSize*(communicatorSize-1)));

	MPI_Request sendReq[2];
	int* sliceSizeMsg = new int[2];
	sliceSizeMsg[0] = sliceSize;
	sliceSizeMsg[1] = numModels;
	MPI_Ibcast(sliceSizeMsg, 2, MPI_INT, 0, MPI_COMM_WORKERS, &sendReq[0]);
	int* sizes = new int[communicatorSize-1];
	int* displs = new int[communicatorSize-1];
	sizes[0] = sliceSize+rowsLeftOver;
	displs[0] = 0;
	displs[1] = sliceSize+rowsLeftOver;
	for(int i= 1;i<communicatorSize-1;i++)
	{
		sizes[i] = sliceSize;
	}
	for(int i = 2; i<communicatorSize-1;i++)
	{
		displs[i] = displs[i-1]+sliceSize;
	}
	Data* myRows = new Data[sliceSize+rowsLeftOver];
	MPI_Scatterv(allRows, sizes, displs, dataStruct, myRows, sliceSize+rowsLeftOver, dataStruct, 0, MPI_COMM_WORKERS);
	processRows(0, MPI_COMM_WORKERS, myRows, sliceSize+rowsLeftOver, numModels, dataStruct, yearToReport, typeToReport);
}

void ranki(int rank, int yearToReport, char typeToReport)
{
	MPI_Comm MPI_COMM_WORKERS;
	MPI_Comm_split(MPI_COMM_WORLD, 0, rank, &MPI_COMM_WORKERS);

	MPI_Datatype dataStruct;
	defineDataStruct(&dataStruct);

	//recieve num rows
	int* sliceSizeMsg = new int[2];
	MPI_Request dataReq[2];
	MPI_Ibcast(sliceSizeMsg, 2, MPI_INT, 0, MPI_COMM_WORKERS, &dataReq[0]);
	MPI_Status recvStatus[2];
	MPI_Wait(&dataReq[0], &recvStatus[0]);
	int sliceSize = sliceSizeMsg[0];
	int numModels = sliceSizeMsg[1];
	if(sliceSize == -1 || numModels==-1) return;
	Data* rows = new Data[sliceSize];
	MPI_Scatterv(nullptr, nullptr, nullptr, dataStruct, rows, sliceSize, dataStruct, 0, MPI_COMM_WORKERS);
//	cout << rows[sliceSize-1].amount << endl;
	processRows(rank, MPI_COMM_WORKERS, rows, sliceSize, numModels, dataStruct, yearToReport, typeToReport);

}
void errorHandlerRank(int rank)
{
	MPI_Comm MPI_COMM_ERROR;
	MPI_Comm_split(MPI_COMM_WORLD, 1, 0, &MPI_COMM_ERROR);

	MPI_Datatype dataStruct;
	defineDataStruct(&dataStruct);
	bool done = false;
	int messageWaiting = 0;
	Data* recvBuf = new Data[1];
	MPI_Request recvReq;
	MPI_Status recvStatus;
	int numErrors = 0;
	while(true)
	{
		if(done){
			//error messages might still be waiting when rank0 terminates
			//errorhandler won't return until Iprobe returns false - no messages waiting
			MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &messageWaiting, &recvStatus);
			if(messageWaiting==0)
			{
				cout << "Total errors: " << numErrors << endl;
				return;
			}
		}
		MPI_Irecv(recvBuf, 1, dataStruct, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &recvReq);
		MPI_Wait(&recvReq, &recvStatus);
		numErrors++;
		int src = recvStatus.MPI_SOURCE;
		Data errorData = recvBuf[0];
		string errorString = "";
		switch(recvStatus.MPI_TAG)
		{
			case 0:
				errorString = "Invalid model number.";
				break;
			case 1:
				errorString = "Invalid date length.";
				break;
			case 2:
				errorString = "Invalid month.";
				break;
			case 3:
				errorString = "Date outside of range.";
				break;
			case 4:
				errorString = "Invalid customer type.";
				break;
			case 5:
				errorString = "Zero or negative sale amount.";
				break;
			case 6:
				done = true;
				numErrors--;
				break;
			}
		if(recvStatus.MPI_TAG != 6) cout << "Rank " << src << " reported error: \"" << errorString << "\": " << dataString(errorData) << endl;

	}
}

int main(int argc, char *argv[])
{
	MPI_Init(&argc, &argv);
	int rank, communicatorSize;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &communicatorSize);
	if(rank == 0) 
	{
		rank0(communicatorSize, argv[1], stoi(argv[2]), argv[3][0]);
	}
	else if (rank==1) errorHandlerRank(1);
	else ranki(rank, stoi(argv[2]), argv[3][0]);
	if(rank!=0)
	{
		MPI_Barrier(MPI_COMM_WORLD);
	}
	MPI_Finalize();
	return 0;
}

