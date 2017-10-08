#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <sstream>
#include <cstdlib>
#include <cstdio>
#include "Seal/seal.h"
#include "Seal/plaintext.h"
#include "Seal/ciphertext.h"
#include "Seal/encryptor.h"
#include "Seal/encryptionparams.h"
#include "Seal/keygenerator.h"
#include "Seal/evaluationkeys.h"
#include "Seal/bigpoly.h"
#include "Seal/biguint.h"
#include "Seal/bigpolyarray.h"
#include "Seal/chooser.h"
#include "Seal/evaluator.h"

using namespace std;

class OwnLinearRegression {
private:
	// compute the cost function J(theta)
	seal::Ciphertext compute_cost(int n_row, int n_col, seal::Ciphertext **x,
			seal::Ciphertext y[], seal::Ciphertext **theta,
			seal::Evaluator evaluate, seal::Plaintext con, bool ridge,
			seal::Plaintext lambda_div) {
		//calculate predictions

		seal::Ciphertext *predictions = calculate_predictions(n_row, n_col, x,
				theta, evaluate);

		// stores difference h(x) -y for each data point
		seal::Ciphertext diff[n_col];
		//stores square of diff for each data point
		seal::Ciphertext sq_errors[n_col];
		//compute (h(x)-y)^2
		for (int i = 0; i < n_col; i++) {

			diff[i] =
					seal::Ciphertext(
							evaluate.sub(
									predictions[i].operator const seal::BigPolyArray &(),
									y[i].operator const seal::BigPolyArray &()));

			sq_errors[i] =
					seal::Ciphertext(
							evaluate.relinearize(
									seal::Ciphertext(
											evaluate.square(
													diff[i].operator const seal::BigPolyArray &())).operator const seal::BigPolyArray &()));

		}
		// creates sq_errors as a vector to add all entries of the data at once
		// additional computations for ridge regression
		if (ridge) {
			vector<seal::Ciphertext> rid;
			for (int i = 0; i <= n_row; i++) {
				seal::Ciphertext rid_square =
						seal::Ciphertext(
								evaluate.relinearize(
										seal::Ciphertext(
												evaluate.square(
														theta[i][0].operator const seal::BigPolyArray &()))));
				rid.emplace_back(rid_square);
			}
			seal::Ciphertext regularizer_tmp = evaluate.add_many(rid);
			seal::Ciphertext regularizer =
					seal::Ciphertext(
							evaluate.multiply_plain(
									regularizer_tmp.operator const seal::BigPolyArray &(),
									lambda_div.operator const seal::BigPoly &()));
			for (int i = 0; i < n_col; i++) {
				sq_errors[i] =
						seal::Ciphertext(
								evaluate.add(
										sq_errors[i].operator const seal::BigPolyArray &(),
										regularizer.operator const seal::BigPolyArray &()));
			}
		}
// end of additional computations of ridge regression

		vector<seal::Ciphertext> t;

		for (int i = 0; i < n_col; i++) {
			t.emplace_back(sq_errors[i]);
		}
//add all
		seal::Ciphertext r = evaluate.add_many(t);

//multiply result by (1.0 / (2 * n_col))
		seal::Ciphertext res = seal::Ciphertext(
				evaluate.multiply_plain(r.operator const seal::BigPolyArray &(),
						con));
		return res;
	}

//helper function that does actual prediction
	seal::Ciphertext h(seal::Ciphertext x[], seal::Ciphertext **theta,
			int n_row, seal::Evaluator evaluate) {
		seal::Ciphertext *res = new seal::Ciphertext[n_row + 1];
// theta_i * x_i
		seal::Ciphertext tmp =
				seal::Ciphertext(
						evaluate.relinearize(
								seal::Ciphertext(
										evaluate.multiply(
												(**theta).operator const seal::BigPolyArray &(),
												x[0].operator const seal::BigPolyArray &())).operator const seal::BigPolyArray &()));
		*res = tmp;
// multiply theta with corresponding x and relinearize
		for (int i = 1; i <= n_row; i++) {
			seal::Ciphertext tmo = seal::Ciphertext(
					evaluate.multiply(
							theta[i][0].operator const seal::BigPolyArray &(),
							x[i].operator const seal::BigPolyArray &()));
			*(res + i) = seal::Ciphertext(
					evaluate.relinearize(
							tmo.operator const seal::BigPolyArray &()));
		}
//create vector of result of the loop theta_0 * x_0, ..., theta_n * x_n
		vector<seal::Ciphertext> t;
		for (int i = 0; i <= n_row; i++) {
			t.emplace_back(res[i]);
		}
// add all ciphertexts
		seal::Ciphertext r = evaluate.add_many(t);

		return r;
	}

// loop over all columns: prediction for one col
	seal::Ciphertext *calculate_predictions(int n_row, int n_col,
			seal::Ciphertext **x, seal::Ciphertext **theta,
			seal::Evaluator evaluate) {
		seal::Ciphertext *predictions = new seal::Ciphertext[n_col];
// calculate h for each training example
		seal::Ciphertext tmp = h(x[0], theta, n_row, evaluate);
		*predictions = tmp;
		for (int i = 1; i < n_col; i++) {
			*(predictions + i) = h(x[i], theta, n_row, evaluate);
		}
		return predictions;
	}

//paramter: pre initialized theta with 1
//perform gradien descent
	seal::Ciphertext **gradient_descent(int n_col, seal::Ciphertext **theta,
			seal::Ciphertext **x, seal::Ciphertext y[], seal::Plaintext alpha,
			int iters, seal::Ciphertext *J, seal::Evaluator evaluate, int n_row,
			seal::Plaintext text, bool ridge, seal::Plaintext lambda_div) {
		seal::Ciphertext ** thet = new seal::Ciphertext*[1];

		for (int i = 0; i < iters; i++) {
			seal::Ciphertext *predictions = calculate_predictions(n_row, n_col,
					x, theta, evaluate);

			// h(x)-y
			seal::Ciphertext diff[n_col];
			for (int j = 0; j < n_col; j++) {
				diff[j] = evaluate.sub(
						predictions[j].operator const seal::BigPolyArray &(),
						y[j].operator const seal::BigPolyArray &());
			}
			seal::Ciphertext res[n_row + 1];
			seal::Ciphertext r[n_row + 1];

			// Plaintext plain = 1/n_row
			for (int k = 0; k <= n_row; k++) {

				//(h(x) -y)* x^j_k
 				thet[k] = new seal::Ciphertext[1];
				for (int j = 0; j < n_col; j++) {
					res[k] =
							seal::Ciphertext(
									evaluate.relinearize(
											seal::Ciphertext(
													evaluate.multiply(
															diff[k].operator const seal::BigPolyArray &(),
															x[j][k].operator const seal::BigPolyArray &())).operator const seal::BigPolyArray &()));

					//if ridge regression add regularizer
					if (ridge && k != 0) {
						seal::Ciphertext rid_tmp =
								seal::Ciphertext(
										evaluate.relinearize(
												seal::Ciphertext(
														evaluate.multiply_plain(
																theta[k][0].operator const seal::BigPolyArray &(),
																lambda_div.operator const seal::BigPoly &())).operator const seal::BigPolyArray &()));
						res[k] =
								seal::Ciphertext(
										evaluate.add(
												res[k].operator const seal::BigPolyArray &(),
												rid_tmp.operator const seal::BigPolyArray &()));
					}
					//(h(x) -y)* x^j_k * alpha

					r[k] =
							seal::Ciphertext(
									evaluate.relinearize(
											seal::Ciphertext(
													evaluate.multiply_plain(
															res[k].operator const seal::BigPolyArray &(),
															alpha)).operator const seal::BigPolyArray &()));
					//((h(x) -y)* x^j_k * alpha)- theta^k_0

					thet[k][0] =
							seal::Ciphertext(
									evaluate.sub(
											theta[k][0].operator const seal::BigPolyArray &(),
											r[k].operator const seal::BigPolyArray &()));

				}

			}

//			J[i] = compute_cost(n_row, n_col, x, y, theta, evaluate, text,
//					ridge, lambda_div);

		}

		return thet;
	}

public:
// train the linear regression model, i.e compute the coefficients theta
	seal::Ciphertext **train(seal::Plaintext alpha, int iterations, int n_col,
			seal::Ciphertext **x, seal::Ciphertext y[], seal::Ciphertext **th,
			int n_row, seal::Plaintext text, seal::Evaluator evaluate,
			bool ridge, seal::Plaintext lambda_div) {
		seal::Ciphertext J[iterations];
		seal::Ciphertext **theta = new seal::Ciphertext *[1];
		seal::Ciphertext **tht = new seal::Ciphertext *[1];
		for (int i = 0; i <= n_row; i++) {
			tht[i] = new seal::Ciphertext[1];
			tht[i][0] = th[i][0];
		}
		//currently not working, assignment fails
		seal::Ciphertext **thet = gradient_descent(n_col, tht, x, y, alpha,
				iterations, J, evaluate, n_row, text, ridge, lambda_div);
		theta = thet;
		return theta;
	}

// predict the target for each data
	seal::Ciphertext *predict(int n_row, int n_col, seal::Ciphertext **x,
			seal::Ciphertext **theta, seal::Evaluator evaluate) {
		seal::Ciphertext *res = new seal::Ciphertext[n_col];

		seal::Ciphertext **tht = new seal::Ciphertext *[1];
		for (int i = 0; i <= n_row; i++) {
			tht[i] = new seal::Ciphertext[1];
			tht[i][0] = theta[i][0];
		}
		// prediction done in helper function
		for (int i = 0; i < n_col; i++) {
			seal::Ciphertext t = h(*(x + i), tht, n_row, evaluate);
			res[i] = t;
		}
		seal::Ciphertext *r = new seal::Ciphertext[n_col];
		r = res;
		return r;
	}
};
