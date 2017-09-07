
#ifndef OWNLINEARREGRESSION_H_
#define OWNLINEARREGRESSION_H_


#include "seal.h"
#include "plaintext.h"
#include "ciphertext.h"
#include "encryptor.h"
#include "evaluator.h"

class OwnLinearRegression {
public:

// trains a linear regression model
	seal::Ciphertext** train(seal::Plaintext alpha, int iterations, int n_col,
			seal::Ciphertext **x, seal::Ciphertext y[], seal::Ciphertext **th,
			int n_row, seal::Plaintext text, seal::Evaluator evaluate,
			bool ridge, seal::Plaintext lambda_div);
// predicts the outcome of a data set
	seal::Ciphertext* predict(int n_row, int n_col, seal::Ciphertext **x,
			seal::Ciphertext **theta, seal::Evaluator evaluate);

private:
	// compuites cost function J
	seal::Ciphertext compute_cost(int n_row, int n_col, seal::Ciphertext **x,
			seal::Ciphertext y[], seal::Ciphertext **theta,
			seal::Evaluator evaluate, seal::Plaintext con, bool ridge,
			seal::Plaintext lambda_div);
	//makes acutal prediction (help function)
	seal::Ciphertext h(seal::Ciphertext x[], seal::Ciphertext **theta,
			int n_col, seal::Evaluator evaluate);
	seal::Ciphertext * calculate_predictions(int n_row, int n_col,
			seal::Ciphertext **x, seal::Ciphertext **theta,
			seal::Evaluator evaluate);

	//performs gradient descent
	seal::Ciphertext ** gradient_descent(int n_col, seal::Ciphertext **&theta,
			seal::Ciphertext **x, seal::Ciphertext y[], seal::Plaintext alpha,
			int iters, seal::Ciphertext *J, seal::Evaluator evaluate, int n_row,
			seal::Plaintext text, bool ridge, seal::Plaintext lambda_div);

};
#endif /* OWNLINEARREGRESSION_H_ */
