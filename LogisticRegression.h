/*
 * LogisticRegression.h
 *
 *  Created on: 25.08.2017
 *      Author: test
 */

#ifndef LOGISTICREGRESSION_H_
#define LOGISTICREGRESSION_H_

//============================================================================
// Name        : LogisticRegression.cpp
// Author      :
// Version     :
// Copyright   :
// Description : Implementation of logistic regression using homomorphic encryption
//============================================================================

#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <sstream>
#include <bitset>
#include <cstdlib>
#include <cstdio>
#include </usr/include/mlpack/core.hpp>
#include </usr/include/mlpack/methods/linear_regression/linear_regression.hpp>
#include "seal.h"
#include "plaintext.h"
#include "ciphertext.h"
#include "encryptor.h"
#include "encryptionparams.h"
#include "keygenerator.h"
#include "evaluationkeys.h"
#include "bigpoly.h"
#include "biguint.h"
#include "bigpolyarray.h"
#include "chooser.h"
#include "evaluator.h"

using namespace std;

class LogisticRegression {
public:
	LogisticRegression();
	LogisticRegression(seal::Ciphertext** theta, seal::Ciphertext** data,
			seal::Ciphertext* target);
	// class 1 if log(h(x)) >= log(thresh) else class 0
	seal::Ciphertext* predict(int col, int row, seal::Ciphertext** data,
			seal::Ciphertext** theta, seal::Plaintext a0, seal::Plaintext a1,
			seal::Plaintext a2, seal::Evaluator evaluate);
	seal::Ciphertext** train(int iters, int col, int row,
			seal::Ciphertext** data, seal::Ciphertext** &theta,
			seal::Ciphertext* target, seal::Evaluator evaluate,
			seal::Plaintext fr, seal::Plaintext a0, seal::Plaintext a1,
			seal::Plaintext a2, seal::Plaintext minus, seal::Plaintext tr,
			seal::Plaintext alpha);
private:
	seal::Ciphertext** gradient_descent(int iters, int col, int row,
			seal::Ciphertext** data, seal::Ciphertext** &theta,
			seal::Ciphertext* target, seal::Evaluator evaluate,
			seal::Plaintext fr, seal::Plaintext a0, seal::Plaintext a1,
			seal::Plaintext a2, seal::Plaintext minus, seal::Plaintext tr,
			seal::Plaintext alpha);
//computes 	a0 -a1*(theta* x) + a2*(theta* x)² for a single data point
	seal::Ciphertext log_h(bool neg, seal::Ciphertext* data,
			seal::Ciphertext** theta, seal::Plaintext a0, seal::Plaintext a1,
			seal::Plaintext a2, seal::Evaluator evaluate, int row, int col);

	seal::Ciphertext compute_A(int j, int col, seal::Ciphertext* targets,
			seal::Ciphertext** data, seal::Ciphertext** theta,
			seal::Evaluator evaluate, seal::FractionalEncoder encoder);
	seal::Ciphertext* compute_helper(int col, int row, seal::Ciphertext** theta,
			seal::Ciphertext** data, seal::Ciphertext* target,
			seal::Plaintext fr, seal::Evaluator evaluate, seal::Plaintext a0,
			seal::Plaintext a1, seal::Plaintext a2, seal::Plaintext minus);
	seal::Ciphertext compute_cost(int col, int row, seal::Ciphertext** theta,
			seal::Ciphertext** data, seal::Ciphertext* target,
			seal::Plaintext fr, seal::Evaluator evaluate, seal::Plaintext a0,
			seal::Plaintext a1, seal::Plaintext a2, seal::Plaintext minus,
			seal::Plaintext tr);
};
#endif /* LOGISTICREGRESSION_H_ */
