**Prerequisites**
To compile and run the programm I use:
*Ubuntu 17.04
*Eclipse Oxygen 
*Gcc version 6.3.0 
Eclipse Oxygen needs to be downloaded from the official website.

**Installation**
**Dependencies**
The programm requires:
*C++ (>= C++11)
*Mlpack
*Boost C++ Libraries
**User installation**
Gcc can be installed from the repositories using "sudo apt-get install gcc".
For further information on how to install mlpack including Boost C++ Libraries see: "http://mccormickml.com/2017/02/01/getting-started-with-mlpack/"

**Running the programm**
I used Eclipse Oxygen as my development environment.
To compile the programm in the command line I used gcc: "g++ *.cpp util/*.cpp -o bt -std=c++11 -larmadillo -lmlpack -lboost_serialization" and executed the resulting script with "./bt".