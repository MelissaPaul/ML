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
#include "seal.h"
#include "plaintext.h"
#include "ciphertext.h"
#include "evaluator.h"

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
	seal::Ciphertext** train(int iters, int col, int row,
			seal::Ciphertext** data, seal::Ciphertext** theta,
			seal::Ciphertext* target, seal::Evaluator evaluate,
			seal::Plaintext fr, seal::Plaintext a0, seal::Plaintext a1,
			seal::Plaintext a2, seal::Plaintext minus, seal::Plaintext tr,
			seal::Plaintext alpha);

private:
    /**
     *
     * @param iters
     * @param col
     * @param row
     * @param data
     * @param theta
     * @param target
     * @param evaluate
     * @param fr
     * @param a0
     * @param a1
     * @param a2
     * @param minus
     * @param tr
     * @param alpha
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
 * @param data
 * @param theta
 * @param a0
 * @param a1
 * @param a2
 * @param evaluate
 * @param row
 * @param col
 * @return
 */
    seal::Ciphertext log_h(bool neg, seal::Ciphertext *data,
                           seal::Ciphertext **theta, seal::Plaintext a0, seal::Plaintext a1,
                           seal::Plaintext a2, seal::Evaluator evaluate, int row, int col);

/**
 *
 * @param j
 * @param col
 * @param targets
 * @param data
 * @param theta
 * @param evaluate
 * @param encoder
 * @return
 */
    seal::Ciphertext compute_A(int j, int col, seal::Ciphertext* targets,
                               seal::Ciphertext** data, seal::Ciphertext** theta,
                               seal::Evaluator evaluate, seal::FractionalEncoder encoder);

    /**
     *
     * @param col
     * @param row
     * @param theta
     * @param data
     * @param target
     * @param fr
     * @param evaluate
     * @param a0
     * @param a1
     * @param a2
     * @param minus
     * @return
     */
    seal::Ciphertext *compute_helper(int col, int row, seal::Ciphertext **theta,
                                     seal::Ciphertext **data, seal::Ciphertext *target,
                                     seal::Plaintext fr, seal::Evaluator evaluate, seal::Plaintext a0,
                                     seal::Plaintext a1, seal::Plaintext a2, seal::Plaintext minus);

    /**
     *
     * @param col
     * @param row
     * @param theta
     * @param data
     * @param target
     * @param fr
     * @param evaluate
     * @param a0
     * @param a1
     * @param a2
     * @param minus
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
