//============================================================================
// Name        : bt.cpp
// Author      :
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================
#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <sstream>
#include <bitset>
#include <cstdlib>
#include <cstdio>
#include "seal.h"
#include "plaintext.h"
#include "ciphertext.h"
#include "encryptor.h"
#include "encryptionparams.h"
#include "keygenerator.h"
#include "bigpoly.h"
#include "biguint.h"
#include "bigpolyarray.h"
#include "chooser.h"
#include "LinearRegression.h"
#include "evaluator.h"
using namespace std;
using namespace seal;

// h_x_i= theta[0] + theta[1] * x_1 + ... + theta[n} * x_2

LinearRegression::LinearRegression(seal::Ciphertext x[][],
		seal::Ciphertext y[], int n_col) {
	this->x = x;
	this->y = y;
	this->n_col = c_col;
}

void LinearRegression ::train(double alpha, int iterations) {
	seal::Ciphertext *J = new seal::Ciphertext[iterations];
	this->theta = gradient_descent(x, y, alpha, iterations, J, n_col);
}

seal::CiphertextLinearRegression::predict(seal::Ciphertext x[],
		seal::Evaluator evaluate) {
	return h(x, theta, n_col, evaluate);
}

seal::CiphertextLinearRegression::compute_cost(seal::Ciphertext x[][],
		seal::Ciphertext y[], seal::Ciphertext theta[], int n_col,
		seal::Evaluator evaluate) {
	seal::Ciphertext *predictions = calculate_predictions(x, theta, n_col,
			evaluate);
	seal::Ciphertext *diff = evaluate.sub(predictions, y);
	seal::Ciphertext *sq_errors = evaluate.square(diff);
	seal::Ciphertext res = sq_errors[0];
	for (int i = 1; i < n_col; i++) {
		res = evaluate.add(res, sq_errors[i]);
	}
	res = evaluate.multiply_plain(res, (1.0 / (2 * n_col)));
	return res;
}

seal::CiphertextLinearRegression::h(seal::Ciphertext xy[],
		seal::Ciphertext thet[], int n_col, seal::Evaluator evaluate) {
	const seal::BigPolyArray theta[] = thet;
	const seal::BigPolyArray x[] = xy;
	seal::BigPolyArray res[];
	evaluate.multiply(theta, x);
	seal::Ciphertext r = res[0];
	for (int i = 1; i < n_col; i++) {
		r = evaluate.add(r, res[i]);
	}
	return r;
}

// loop over all rows: prediction for one row
seal::Ciphertext *LinearRegression::calculate_predictions(
		seal::Ciphertext x[][], seal::Ciphertext theta[], int n_col,
		seal::Evaluator evaluate) {

	seal::Ciphertext predictions[] = new seal::Ciphertext[n_col];

	// calculate h for each training example
	for (int i = 0; i < n_col; i++) {
		predictions[i] = (x[i], theta, n_col, evaluate);
	}
	return predictions;
}
//paramter: pre initialized theta with 1
seal::Ciphertext *LinearRegression::gradient_descent(seal::Ciphertext theta[],
		seal::Ciphertext x[][], seal::Ciphertext y[], double alpha, int iters,
		seal::Ciphertext *J, seal::Evaluator evaluate, int n_row, int n_col) {

	for (int i = 0; i < iters; ++i) {
		seal::Ciphertext *predictions = calculate_predictions(x, theta, n_col,
				evaluate);

		// h(x-y)
		seal::Ciphertext *diff = evaluate.sub(predictions, y);
		seal::Ciphertext res[];

		for (int i = 0; i < n_row; i++) {
			for (int j = 0; j < n_col; j++) {
				res[i] = evaluate.multiply(diff, x[i][j]);
			}
			res = evaluate.multiply_plain(res, 1 / n_row);
			res = evaluate.multiply_plain(res, alpha);
			theta[i] = evaluate.sub(theta[i], res);
		}

		J[i] = compute_cost(x, y, theta, n_row);
	}
	return theta;
}

