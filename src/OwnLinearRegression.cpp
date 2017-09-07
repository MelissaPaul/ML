#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <sstream>
#include <cstdlib>
#include <cstdio>
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
			cout << i << endl;
			diff[i] =
					seal::Ciphertext(
							evaluate.sub(
									predictions[i].operator const seal::BigPolyArray &(),
									y[i].operator const seal::BigPolyArray &()));
			//	cout << "after diff" << endl;
			sq_errors[i] =
					seal::Ciphertext(
							evaluate.relinearize(
									seal::Ciphertext(
											evaluate.square(
													diff[i].operator const seal::BigPolyArray &())).operator const seal::BigPolyArray &()));
			//	cout << "sq" << sq_errors[i].size() << endl;
			//cout << "after sq" << endl;
		}
		// creates sq_errors as a vector to add all entries of the data at once
		if (ridge) {
			vector<seal::Ciphertext> rid;
			for (int i = 0; i <= n_row; i++) {
				cout << i << "in ridge" << endl;
				seal::Ciphertext rid_square =
						seal::Ciphertext(
								evaluate.relinearize(
										seal::Ciphertext(
												evaluate.square(
														theta[i][0].operator const seal::BigPolyArray &()))));
				rid.emplace_back(rid_square);
			}
			cout << "after in ridge" << endl;
			seal::Ciphertext regularizer_tmp = evaluate.add_many(rid);
			seal::Ciphertext regularizer =
					seal::Ciphertext(
							evaluate.multiply_plain(
									regularizer_tmp.operator const seal::BigPolyArray &(),
									lambda_div.operator const seal::BigPoly &()));
			cout << "before sqaure in ridge" << endl;
			for (int i = 0; i < n_col; i++) {
				cout << i << "sqer" << endl;
				sq_errors[i] =
						seal::Ciphertext(
								evaluate.add(
										sq_errors[i].operator const seal::BigPolyArray &(),
										regularizer.operator const seal::BigPolyArray &()));
			}
			cout << "after ridge" << endl;
		}

		vector<seal::Ciphertext> t;

		for (int i = 0; i < n_col; i++) {
			t.emplace_back(sq_errors[i]);
		}
//add all
		seal::Ciphertext r = evaluate.add_many(t);
//cout << "add finished" << endl;
//multiply result by (1.0 / (2 * n_col))
		seal::Ciphertext res = seal::Ciphertext(
				evaluate.multiply_plain(r.operator const seal::BigPolyArray &(),
						con));
//cout << "multiply done" << endl;
		return res;
	}

//helper function that does actual prediction
	seal::Ciphertext h(seal::Ciphertext x[], seal::Ciphertext **theta,
			int n_row, seal::Evaluator evaluate) {
		seal::Ciphertext* res = new seal::Ciphertext[n_row + 1];
//	cout << "inside h " << endl;
		/*
		 seal::Ciphertext tmp = seal::Ciphertext(
		 evaluate.multiply((**theta).operator const seal::BigPolyArray &(),
		 x[0].operator const seal::BigPolyArray &()));
		 */
//	cout << "0 before" << (**theta).operator seal::BigPolyArray &().size()
//			<< " and " << x[0].operator seal::BigPolyArray &().size() << endl;
//cout << x[0].operator seal::BigPolyArray &().size() << endl;
		seal::Ciphertext tmp =
				seal::Ciphertext(
						evaluate.relinearize(
								seal::Ciphertext(
										evaluate.multiply(
												(**theta).operator const seal::BigPolyArray &(),
												x[0].operator const seal::BigPolyArray &())).operator const seal::BigPolyArray &()));
//cout << "tmp works" << endl;
//cout << "after" << tmp.size() << endl;
		*res = tmp;
//	cout << "first multiplz worked " << endl;
// multiply theta with corresponding x
		for (int i = 1; i <= n_row; i++) {
			//	cout << i << endl;
			//	cout << theta[i][0].operator seal::BigPolyArray &().coeff_count()
			//			<< "  "
			//			<< theta[i][0].operator seal::BigPolyArray &().coeff_bit_count()
			//			<< endl;
			seal::Ciphertext tmo = seal::Ciphertext(
					evaluate.multiply(
							theta[i][0].operator const seal::BigPolyArray &(),
							x[i].operator const seal::BigPolyArray &()));
			//	cout << "multiply fine" << endl;
			*(res + i) = seal::Ciphertext(
					evaluate.relinearize(
							tmo.operator const seal::BigPolyArray &()));
//		cout << res[i].operator seal::BigPolyArray &().coeff_count() << " in h "
//				<< res[i].operator seal::BigPolyArray &().coeff_bit_count()
//				<< endl;
//
			/*seal::Ciphertext(
			 evaluate.multiply(
			 theta[i][0].operator const seal::BigPolyArray &(),
			 x[i].operator const seal::BigPolyArray &()));*/
//		seal::Ciphertext tmp = *(res + i);
			//	cout << tmp.size() << endl;
			//		cout << "multiplyinhg" << endl;
		}
//create vector of result of the loop theta_0 * x_0, ..., theta_n * x_n
		vector<seal::Ciphertext> t;
//	cout << "vector created" << endl;
		for (int i = 0; i <= n_row; i++) {
			t.emplace_back(res[i]);
		}
//	cout << "Vector filled" << endl;
// add all ciphertexts
//cout << "adding" << endl;
		seal::Ciphertext r = evaluate.add_many(t);
//	cout << "adding done" << endl;

		return r;
	}

// loop over all rows: prediction for one row
	seal::Ciphertext * calculate_predictions(int n_row, int n_col,
			seal::Ciphertext **x, seal::Ciphertext **theta,
			seal::Evaluator evaluate) {
//	cout << "caculat oredict"
//			<< x[0][0].operator seal::BigPolyArray &().coeff_count() << " "
//			<< x[0][0].operator seal::BigPolyArray &().coeff_bit_count()
//			<< endl;
		seal::Ciphertext* predictions = new seal::Ciphertext[n_col];
//	cout << "inside calculate pred" << endl;
// calculate h for each training example
		seal::Ciphertext tmp = h(x[0], theta, n_row, evaluate);
//	cout << "first one worked" << endl;
		*predictions = tmp;
//	cout << "allocation worked" << endl;
		for (int i = 1; i < n_col; i++) {
			//	cout << i << endl;
			//	Ciphertext* tmp = get_column(x, i, n_row, n_col);
			//	cout<< "getcolumn works for "
			*(predictions + i) = h(x[i], theta, n_row, evaluate);
			//	cout << "compute h" << endl;
//		cout << predictions[i].operator seal::BigPolyArray &().coeff_count()
//				<< " in calc "
//				<< predictions[i].operator seal::BigPolyArray &().coeff_bit_count()
//				<< endl;
		}
		return predictions;
	}
//paramter: pre initialized theta with 1
//perform gradien descent
	seal::Ciphertext ** gradient_descent(int n_col, seal::Ciphertext **&theta,
			seal::Ciphertext **x, seal::Ciphertext y[], seal::Plaintext alpha,
			int iters, seal::Ciphertext *J, seal::Evaluator evaluate, int n_row,
			seal::Plaintext text, bool ridge, seal::Plaintext lambda_div) {
		//	cout << "in gradient descent" << endl;

//		for (int i = 0; i < n_col; i++) {
//			cout << y[i].operator seal::BigPolyArray &().coeff_count() << endl;
//			cout << y[i].operator seal::BigPolyArray &().coeff_bit_count()
//					<< endl;
//		}
		for (int i = 0; i < iters; i++) {
			cout << i << endl;
			seal::Ciphertext *predictions = calculate_predictions(n_row, n_col,
					x, theta, evaluate);
			//	cout << "made prediction " << i << endl;
			// h(x)-y
			seal::Ciphertext diff[n_col];
			for (int j = 0; j < n_col; j++) {
				cout << j << endl;
				diff[j] = evaluate.sub(
						predictions[j].operator const seal::BigPolyArray &(),
						y[j].operator const seal::BigPolyArray &());
				//	cout << "calculated difference " << j << endl;
			}
			//		cout << "fine" << endl;
			seal::Ciphertext res[n_row + 1];
			seal::Ciphertext r[n_row + 1];
			// Plaintext plain = 1/n_row
			for (int k = 0; k <= n_row; k++) {
				for (int j = 0; j < n_col; j++) {
					res[k] =
							seal::Ciphertext(
									evaluate.relinearize(
											seal::Ciphertext(
													evaluate.multiply(
															diff[k].operator const seal::BigPolyArray &(),
															x[j][k].operator const seal::BigPolyArray &())).operator const seal::BigPolyArray &()));
					//			cout << "calculated multiply" << endl;
					//if ridge regression add regularizer
					if (ridge && k != 0) {
						seal::Ciphertext rid_tmp =
								seal::Ciphertext(
										evaluate.multiply_plain(
												theta[k][0].operator const seal::BigPolyArray &(),
												lambda_div.operator const seal::BigPoly &()));
						res[k] =
								seal::Ciphertext(
										evaluate.add(
												res[k].operator const seal::BigPolyArray &(),
												rid_tmp.operator const seal::BigPolyArray &()));
					}
					r[k] =
							seal::Ciphertext(
									evaluate.relinearize(
											seal::Ciphertext(
													evaluate.multiply_plain(
															res[k].operator const seal::BigPolyArray &(),
															alpha)).operator const seal::BigPolyArray &()));
					//		cout << "calculated multiply plain" << endl;
					//cout << r[k].size() << endl;
					theta[k][0] =
							seal::Ciphertext(
									evaluate.sub(
											theta[k][0].operator const seal::BigPolyArray &(),
											r[k].operator const seal::BigPolyArray &()));
					//			cout << "calculated rest " << k << " + " << j << endl;
				}
			}
			cout << "before compute cost" << endl;
			seal::Ciphertext t = compute_cost(n_row, n_col, x, y, theta,
					evaluate, text, ridge, lambda_div);
			cout << "calculated J " << endl;
			J[i] = t;
		}
		return theta;
	}

public:
// train the linear regression model, i.e compute the coefficients theta
	seal::Ciphertext** train(seal::Plaintext alpha, int iterations, int n_col,
			seal::Ciphertext **x, seal::Ciphertext y[], seal::Ciphertext **th,
			int n_row, seal::Plaintext text, seal::Evaluator evaluate,
			bool ridge, seal::Plaintext lambda_div) {
		seal::Ciphertext J[iterations];
		//seal::Ciphertext** theta= new seal::Ciphertext*[1];
		//	cout << "init all right" << endl;
		seal::Ciphertext **theta = new seal::Ciphertext*[1];
		seal::Ciphertext** tht = new seal::Ciphertext*[1];
		for (int i = 0; i <= n_row; i++) {
			tht[i] = new seal::Ciphertext[1];
			//	theta[i] = new seal::Ciphertext[1];
			tht[i][0] = th[i][0];
		}
		seal::Ciphertext **thet = gradient_descent(n_col, tht, x, y, alpha,
				iterations, J, evaluate, n_row, text, ridge, lambda_div);
		//	cout << "train works fine" << endl;
		theta = thet;
		return theta;
	}

// predict the target for each data
	seal::Ciphertext* predict(int n_row, int n_col, seal::Ciphertext **x,
			seal::Ciphertext **theta, seal::Evaluator evaluate) {
		seal::Ciphertext *res = new seal::Ciphertext[n_col];
		//fill_n(res, n_col, new seal::Ciphertext());
//	for (int i = 0; i < n_col; i++) {
//		res[i] = new seal::Ciphertext();
//	}
//	seal::Ciphertext tmp = h(*x, theta, n_row, evaluate);
//	cout << "start predict" << endl;
//	res[0] = tmp;
		seal::Ciphertext** tht = new seal::Ciphertext*[1];
		for (int i = 0; i <= n_row; i++) {
			tht[i] = new seal::Ciphertext[1];
			//	theta[i] = new seal::Ciphertext[1];
			tht[i][0] = theta[i][0];
		}
		for (int i = 0; i < n_col; i++) {
			//cout << i << endl;
			seal::Ciphertext t = h(*(x + i), tht, n_row, evaluate);
			//	cout << "h works" << endl;
			res[i] = t;
			//	cout << "res works" << endl;
//		cout << res[i].operator seal::BigPolyArray &().coeff_count()
//				<< " predict "
//				<< res[i].operator seal::BigPolyArray &().coeff_bit_count()
//				<< endl;

		}
		seal::Ciphertext *r = new seal::Ciphertext[n_col];
		r = res;
//	cout << "prediction works fine" << endl;
		return r;
	}
};
