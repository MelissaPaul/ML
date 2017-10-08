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
#include "Seal/seal.h"
#include "Seal/plaintext.h"
#include "Seal/ciphertext.h"
#include "Seal/evaluator.h"

using namespace std;

class LogisticRegression {
public:

	LogisticRegression();

	LogisticRegression(seal::Ciphertext** theta, seal::Ciphertext** data,
			seal::Ciphertext* target);
	// class 1 if log(h(x)) >= log(thresh) else class 0
	/**
	 *
	 * @param col: # of columns
	 * @param row: # of rows
	 * @param data: encrypted data (two dimensional array)
	 * @param theta: encrypted weights (two dimensional array)
	 * @param a0:
	 * @param a1:
	 * @param a2:
	 * @param evaluate: seal::Evaluator
	 * @return
	 */
	seal::Ciphertext *predict(int col, int row, seal::Ciphertext **data,
			seal::Ciphertext** theta, seal::Plaintext a0, seal::Plaintext a1,
			seal::Plaintext a2, seal::Evaluator evaluate);

	/**
		 * @param classes: # of classes to be considered
		 * @param col: # of columns
		 * @param row: # of rows
		 * @param data: encrypted data (two dimensional array)
		 * @param theta: encrypted weights (two dimensional array)
		 * @param a0:
		 * @param a1:
		 * @param a2:
		 * @param evaluate: seal::Evaluator
		 * @return
		 */
		seal::Ciphertext *predict_multiclass(int classes, int col, int row, seal::Ciphertext **data,
				seal::Ciphertext** theta, seal::Plaintext a0, seal::Plaintext a1,
				seal::Plaintext a2, seal::Evaluator evaluate);

	/**
	 *
	 * @param iters: # of iterations for gradient descent
	 * @param col: # of columns
	 * @param row: # of rows
	 * @param data: encrypted data (two dimensional array)
	 * @param theta: encrypted weights (two dimensional array)
	 * @param target: encrypted targets (one dimensional array)
	 * @param evaluate: seal::Evaluator
	 * @param fr
	 * @param a0
	 * @param a1
	 * @param a2
	 * @param minus: -1 encoded as seal::Plaintext
	 * @param tr
	 * @param alpha: learning rate as double
	 * @return
	 */
	seal::Ciphertext** train(int iters, int col, int row,
			seal::Ciphertext** data, seal::Ciphertext** theta,
			seal::Ciphertext* target, seal::Evaluator evaluate,
			seal::Plaintext fr, seal::Plaintext a0, seal::Plaintext a1,
			seal::Plaintext a2, seal::Plaintext minus, seal::Plaintext tr,
			seal::Plaintext alpha);

private:

	/**
	 *
	 * @param iters: # of iterations for gradient descent
	 * @param col: # of columns
	 * @param row: # of rows
	 * @param data: encrypted data (two dimensional array)
	 * @param theta: encrypted weights (two dimensional array)
	 * @param target: encrypted targets (one dimensional array)
	 * @param evaluate: seal::Evaluator
	 * @param fr
	 * @param a0
	 * @param a1
	 * @param a2
	 * @param minus: -1 encoded as seal::Plaintext
	 * @param tr
	 * @param alpha: learning rate as double
	 * @return
	 */
	seal::Ciphertext **gradient_descent(int iters, int col, int row,
			seal::Ciphertext **data, seal::Ciphertext **theta,
			seal::Ciphertext *target, seal::Evaluator evaluate,
			seal::Plaintext fr, seal::Plaintext a0, seal::Plaintext a1,
			seal::Plaintext a2, seal::Plaintext minus, seal::Plaintext tr,
			seal::Plaintext alpha);
//computes 	a0 -a1*(theta* x) + a2*(theta* x)² for a single data point
	/**
	 *
	 * @param neg
	 * @param data: one row of the encrypted data (one dimensional array)
	 * @param theta: encrypted weights (two dimensional array)
	 * @param a0
	 * @param a1
	 * @param a2
	 * @param evaluate: seal::Evaluator
	 * @param row: # of rows
	 * @param col: # of columns
	 * @return
	 */
	seal::Ciphertext log_h(bool neg, seal::Ciphertext *data,
			seal::Ciphertext **theta, seal::Plaintext a0, seal::Plaintext a1,
			seal::Plaintext a2, seal::Evaluator evaluate, int row, int col);

	/**
	 *
	 * @param j
	 * @param col: # of columns
	 * @param targets: encrypted targets (one dimensional array)
	 * @param data: encrypted data (two dimensional array)
	 * @param theta: encrypted weights (two dimensional array)
	 * @param evaluate: seal::Evaluator
	 * @param encoder: seal::FractionalEncoder
	 * @return
	 */
	seal::Ciphertext compute_A(int j, int col, seal::Ciphertext* targets,
			seal::Ciphertext** data, seal::Ciphertext** theta,
			seal::Evaluator evaluate, seal::FractionalEncoder encoder);

	/**
	 *
	 * @param col: # of columns
	 * @param row: # of rows
	 * @param theta: encrypted weights (two dimensional array)
	 * @param data: encrypted data (two dimensional array)
	 * @param target: encrypted targets (one dimensional array)
	 * @param fr
	 * @param evaluate: seal::Evaluator
	 * @param a0
	 * @param a1
	 * @param a2
	 * @param minus: -1 encoded as seal::Plaintext
	 * @return
	 */
	seal::Ciphertext *compute_helper(int col, int row, seal::Ciphertext **theta,
			seal::Ciphertext **data, seal::Ciphertext *target,
			seal::Plaintext fr, seal::Evaluator evaluate, seal::Plaintext a0,
			seal::Plaintext a1, seal::Plaintext a2, seal::Plaintext minus);

	/**
	 *
	 * @param col: # of columns
	 * @param row: # of rows
	 * @param theta: encrypted weights (two dimensional array)
	 * @param data: encrypted data (two dimensional array)
	 * @param target: encrypted targets (one dimensional array)
	 * @param fr
	 * @param evaluate: seal::Evaluator
	 * @param a0
	 * @param a1
	 * @param a2
	 * @param minus: -1 encoded as seal::Plaintext
	 * @param tr
	 * @return
	 */
	seal::Ciphertext compute_cost(int col, int row, seal::Ciphertext **theta,
			seal::Ciphertext **data, seal::Ciphertext *target,
			seal::Plaintext fr, seal::Evaluator evaluate, seal::Plaintext a0,
			seal::Plaintext a1, seal::Plaintext a2, seal::Plaintext minus,
			seal::Plaintext tr);
};
#endif /* LOGISTICREGRESSION_H_ */
