#pragma once
#ifndef LINEARREGRESSION_H
#define LINEARREGRESSION_H
#include "seal.h"
#include "ciphertext.h"
#include "bigpolyarray.h"

class LinearRegression {
public:

	// First feature
	seal::Ciphertext **x;

	// Target feature
	seal::Ciphertext *y;

	// Number of training examples
	static const int n_col;

	// The theta coefficients
	seal::Ciphertext *theta;

	/**
	 * Create a new instance from the given data set.
	 */
	LinearRegression(seal::Ciphertext x[][n_col], seal::Ciphertext y[],
			int n_col);

	/**
	 * Train the model with the supplied parameters.
	 *
	 * @param alpha         The learning rate, e.g. 0.01.
	 * @param iterations    The number of gradient descent steps to do.
	 */
	void train(double alpha, int iterations);

	/**
	 * Try to predict y, given an x.
	 */
	seal::Ciphertext predict(seal::Ciphertext x);

private:

	/**
	 * Compute the cost J.
	 */
	static seal::Ciphertext compute_cost(seal::Ciphertext x[][n_col],
			seal::Ciphertext y[], seal::Ciphertext theta[], int n_col,
			seal::Evaluator evaluate);
	/**
	 * Compute the hypothesis.
	 */
	static seal::Ciphertext h(seal::Ciphertext x[][n_col],
			seal::Ciphertext theta[], int n_col, seal::Evaluator evaluate);

	/**
	 * Calculate the target feature from the other ones.
	 */
	static seal::Ciphertext *calculate_predictions(seal::Ciphertext x[][n_col],
			seal::Ciphertext theta[], int n_col, seal::Evaluator evaluate);

	/**
	 * Performs gradient descent to learn theta by taking num_items gradient steps with learning rate alpha.
	 */
	static seal::Ciphertext *gradient_descent(seal::Ciphertext theta[],
			seal::Ciphertext x[][n_col], seal::Ciphertext y[], double alpha,
			int iters, seal::Ciphertext *J, seal::Evaluator evaluate, int n_row,
			int n_col);

};

#endif
