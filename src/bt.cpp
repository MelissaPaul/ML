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
#include <cstdlib>
#include <cstdio>
#include </usr/include/mlpack/core.hpp>
#include <math.h>
//#include </usr/include/mlpack/methods/linear_regression/linear_regression.hpp>
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
#include <chrono>
//#include "LogisticRegression.h"
// error with constructor when using header file
#include "LogisticRegression.cpp"
//#include "OwnLinearRegression.h"
#include "OwnLinearRegression.cpp"

//TODO inclusion of header files; deletion of unsued arrays, error: corrputed size vs. previous size in training methods as well as prediction in multiclass classification

using namespace std;
using namespace seal;
using namespace mlpack;
//using namespace mlpack::regression;

//encrypts an integer by turning it into Plaintext (polynomial) and then Ciphertext
Ciphertext encrypt(const int input, Encryptor encryptor,
		IntegerEncoder encoder) {
	Plaintext encoded = encoder.encode(input);
	Ciphertext encrypted = encryptor.encrypt(encoded);
	return encrypted;
}

// encrypt a double
Ciphertext encrypt_frac(double input, Encryptor encryptor,
		FractionalEncoder encoder) {
	Plaintext encoded = encoder.encode(input);
	Ciphertext encrypted = encryptor.encrypt(encoded);
	return encrypted;
}

// decrypts given ciphertext by turning it into Plaintext and then integer
int decrypt(Ciphertext encrypted, Decryptor decryptor, IntegerEncoder encoder) {
	cout << "decryptor" << endl;
	BigPoly decrypted = decryptor.decrypt(
			encrypted.operator const seal::BigPolyArray &());
	int decryptt = encoder.decode_int32(decrypted);
	cout << "int" << endl;
	return decryptt;
}

// decrypt given Ciphertext to double
double decrypt_frac(Ciphertext encrypted, Decryptor decryptor,
		FractionalEncoder encoder) {
	//cout << "decryptor" << endl;
	BigPoly decrypted = decryptor.decrypt(
			encrypted.operator const seal::BigPolyArray &());
//	cout << "Plaintext" << endl;
	double decryptt = encoder.decode(decrypted);
//	cout << "int" << endl;
	return decryptt;
	// Decode results.
}

////perform linear regression

//arma::vec reg_lin_reg(arma::mat data, arma::vec responses, double lambda) {
//
////	// perform linear regression; lambda = 0.5 for ridge regression; 0.0 for linear regression
//	LinearRegression lr(data, responses, lambda);
//	arma::vec parameters = lr.Parameters();
////
//	arma::mat train_data = data.submat(0, 0, 3, 11);
//	arma::vec train_target = responses.subvec(0, 11);
//	arma::mat test_data = data.submat(0, 11, 3, 22);
//	arma::vec test_target = responses.subvec(11, 22);
////	lr.Train(train_data, train_target, false, );
//	arma::vec predictions;
//	lr.Predict(data, predictions);
//	return predictions;
//}

//Ciphertext* reg_lin_reg_enc(Ciphertext[][23] encoded, double lambda) {
//	//	// perform linear regression; lambda = 0.5 for ridge regression; 0.0 for linear regression
//	LinearRegression lr(data, responses, lambda);
//	arma::vec parameters = lr.Parameters();
//	//
//	arma::mat train_data = data.submat(0, 0, 3, 11);
//	arma::vec train_target = responses.subvec(0, 11);
//	arma::mat test_data = data.submat(0, 11, 3, 22);
//	arma::vec test_target = responses.subvec(11, 22);
//	//	lr.Train(train_data, train_target, false, );
//	arma::vec predictions;
//	lr.Predict(data, predictions);
//	return predictions;
//}

/*int* string2ascii(string input) {
 int* ptr = new int[input.length() + 1];
 for (int i = 0; i < input.length(); i++) {
 ptr[i] = (int) input[i];
 }
 ptr[input.length()] = -1;
 return ptr;
 }

 string ascii2string(int *ptr) {
 int length = sizeof(ptr) / sizeof(ptr[0]);
 stringstream ss;
 while (*(ptr) != -1) {
 ss << (char) *(ptr);
 cout << (char) *(ptr) << endl;
 ptr++;
 }
 return ss.str();
 }*/

//uncomment when running binary logistic regression///////
void fun1() {

	//set parameters for encryption
	seal::EncryptionParameters parms;
	parms.set_poly_modulus("1x^4096 + 1");
	parms.set_coeff_modulus(
			seal::ChooserEvaluator::default_parameter_options().at(4096));
	parms.set_plain_modulus(1 << 8);
	parms.set_decomposition_bit_count(58);
	parms.validate();

	// logistic regression
	//read in data for classification
	arma::mat train_dat_reg;
	arma::vec resp_train_reg;
	arma::mat test_dat_reg;
	arma::vec resp_test_reg;
	data::Load("spect_train.csv", train_dat_reg);
	data::Load("spect_test.csv", test_dat_reg);
	arma::mat tmp_reg = train_dat_reg.t();
	arma::mat tmp2_reg = test_dat_reg.t();
	resp_train_reg = tmp_reg.col(0);
	resp_test_reg = tmp2_reg.col(0);
	train_dat_reg.shed_row(0);
	test_dat_reg.shed_row(0);

	//number of columns and rows for training and test set
	int train_col_reg = static_cast<int>(train_dat_reg.n_cols);
	int train_row_reg = static_cast<int>(train_dat_reg.n_rows);
	int test_col_reg = static_cast<int>(test_dat_reg.n_cols);
	int test_row_reg = static_cast<int>(test_dat_reg.n_rows);

//	//	transpose data matrices
//	double **train_dat_trans_reg = new double*[train_row_reg + 1];
//	double ** test_dat_trans_reg = new double*[test_row_reg + 1];
//	for (int i = 0; i < train_col_reg; i++) {
//		train_dat_trans_reg[i] = new double[train_row_reg + 1];
//		train_dat_trans_reg[i][0] = 1;
//		for (int j = 1; j <= train_row_reg; j++) {
//			train_dat_trans_reg[i][j] = train_dat_reg(j - 1, i);
//		}
//	}
//	for (int i = 0; i < test_col_reg; i++) {
//		test_dat_trans_reg[i] = new double[test_row_reg + 1];
//		test_dat_trans_reg[i][0] = 1;
//		for (int j = 1; j <= test_row_reg; j++) {
//			test_dat_trans_reg[i][j] = test_dat_reg(j - 1, i);
//		}
//	}

	//Generate Keys
	seal::KeyGenerator generator(parms);
	generator.generate();
	const seal::BigPolyArray p_key = generator.public_key();
	seal::Ciphertext public_key = seal::Ciphertext(p_key);
	const seal::BigPoly s_key = generator.secret_key();
	seal::Plaintext secret_key = seal::Plaintext(s_key);
	generator.generate_evaluation_keys(test_col_reg);
	const seal::EvaluationKeys evkey = generator.evaluation_keys();
	seal::FractionalEncoder frencoder(parms.plain_modulus(),
			parms.poly_modulus(), 64, 32, 3);
	seal::IntegerEncoder encoder(parms.plain_modulus());
	seal::Evaluator evaluate(parms, evkey);
	seal::Encryptor encryptor(parms, p_key);
	seal::Decryptor decryptor(parms, s_key);

	//#columns, dataset, theta, Evaluator

	seal::Ciphertext ** encoded_train_reg = new seal::Ciphertext*[train_col_reg];
	seal::Ciphertext ** encoded_test_reg = new seal::Ciphertext*[test_col_reg];
	seal::Ciphertext* test_resp_reg = new seal::Ciphertext[test_col_reg];
	seal::Ciphertext* train_resp_reg = new seal::Ciphertext[train_col_reg];

	auto start = chrono::steady_clock::now();
	//	encrypt targets y (1 x columns)
	for (int i = 0; i < train_col_reg; i++) {
		Ciphertext t = encrypt_frac(resp_train_reg(i), encryptor, frencoder);
		train_resp_reg[i] = t;
	}
	auto end = chrono::steady_clock::now();
	auto diffr = end - start;
	cout << "encrypt train targets "
			<< chrono::duration<double, ratio<1>>(diffr).count() << " s"
			<< endl;

	start = chrono::steady_clock::now();
	for (int i = 0; i < test_col_reg; i++) {
		test_resp_reg[i] = encrypt_frac(resp_test_reg(i), encryptor, frencoder);
	}
	end = chrono::steady_clock::now();
	diffr = end - start;
	cout << "encrypt test targets "
			<< chrono::duration<double, ratio<1>>(diffr).count() << " s"
			<< endl;

	start = chrono::steady_clock::now();
	seal::Ciphertext one_encrypt = encrypt_frac(1.0, encryptor, frencoder);
	for (int i = 0; i < train_col_reg; i++) {
		encoded_train_reg[i] = new seal::Ciphertext[train_row_reg + 1];
		encoded_train_reg[i][0] = one_encrypt;
		for (int j = 1; j <= train_row_reg; j++) {
			encoded_train_reg[i][j] = encrypt_frac(train_dat_reg(j - 1, i),
					encryptor, frencoder);
		}
	}
	end = chrono::steady_clock::now();
	diffr = end - start;
	cout << "encrypt train data "
			<< chrono::duration<double, ratio<1>>(diffr).count() << " s"
			<< endl;

	// deleting unused transpose of train matrix
	// deleting of Ciphertext arrays is not working anywhere at the moment
	// for (int i = 0; i <= train_col_reg; i++) {
	// delete[] train_dat_trans_reg[i];
	// }
	//	delete[] train_dat_trans_reg;
	start = chrono::steady_clock::now();

	for (int i = 0; i < test_col_reg; i++) {
		encoded_test_reg[i] = new seal::Ciphertext[test_row_reg + 1];
		encoded_test_reg[i][0] = one_encrypt;
		for (int j = 1; j <= test_row_reg; j++) {
			encoded_test_reg[i][j] = encrypt_frac(test_dat_reg(j - 1, i),
					encryptor, frencoder);
		}
	}
	end = chrono::steady_clock::now();
	diffr = end - start;
	cout << "encrypt test data "
			<< chrono::duration<double, ratio<1>>(diffr).count() << " s"
			<< endl;

	//deleting unused transpose of test matrix
	//	for (int i = 0; i <= test_row_reg; i++) {
	//		delete[] test_dat_trans_reg[i];
	//	}
	//	delete[] test_dat_trans_reg;

	// training rate alpha
	double alph_reg = 0.32;
	// tr = lambda/2*col, lambda security parameter?
	double lambda_reg = 0.4;
	seal::Plaintext fr = frencoder.encode(1 / (1.0 * train_col_reg));
	seal::Plaintext a0 = frencoder.encode(-0.714761);
	seal::Plaintext a1 = frencoder.encode(-0.5);
	seal::Plaintext a2 = frencoder.encode(-0.0976419);
	seal::Plaintext minus_one_reg = frencoder.encode(-1.0);
	seal::Plaintext tr = frencoder.encode(lambda_reg / (2.0 * train_col_reg));
	seal::Plaintext alpha_reg = frencoder.encode(alph_reg);
	LogisticRegression lr; // = LogisticRegression::LogisticRegression();

	// initialize theta for regression and classification with 1 and encrypt it (row x 1)

	seal::Ciphertext **theta_reg = new seal::Ciphertext*[train_row_reg + 1];

	for (int i = 0; i <= train_row_reg; i++) {
		theta_reg[i] = new seal::Ciphertext[1];

		theta_reg[i][0] = one_encrypt;

	}

	// train the classification model
	start = chrono::steady_clock::now();
	seal::Ciphertext** weights_reg = new seal::Ciphertext*[train_row_reg + 1];
	weights_reg = lr.train(1, train_col_reg, train_row_reg, encoded_train_reg,
			theta_reg, train_resp_reg, evaluate, fr, a0, a1, a2, minus_one_reg,
			tr, alpha_reg);
	end = chrono::steady_clock::now();
	diffr = end - start;
	cout << "train" << chrono::duration<double, ratio<60>>(diffr).count()
			<< " min" << endl;

	//delete pre-initialized theta
	//	delete[] theta_reg[0];
	//	delete[] theta_reg;

	start = chrono::steady_clock::now();

	// //make binary class predictions
	seal::Ciphertext* predictions_reg = new seal::Ciphertext[test_col_reg];
	predictions_reg = lr.predict(test_col_reg, test_row_reg, encoded_test_reg,
			weights_reg, a0, a1, a2, evaluate);
	end = chrono::steady_clock::now();
	diffr = end - start;
	cout << "prediction" << chrono::duration<double, ratio<60>>(diffr).count()
			<< " min" << endl;

	//cout << "predict fone" << endl;
	double * predictions_reg_encrypted = new double[test_col_reg];
	int * class_prediction = new int[test_col_reg];
	start = chrono::steady_clock::now();
	for (int i = 0; i < test_col_reg; i++) {
		predictions_reg_encrypted[i] = decrypt_frac(predictions_reg[i],
				decryptor, frencoder);
		if (predictions_reg_encrypted[i] < 0.5) {
			class_prediction[i] = 0;
		} else {
			class_prediction[i] = 1;
		}
		//cout << i << endl;
		cout << class_prediction[i] << endl;
	}
	end = chrono::steady_clock::now();
	diffr = end - start;
	cout << "decrypt test data "
			<< chrono::duration<double, ratio<1>>(diffr).count() << " s"
			<< endl;

}

// uncomment when running multiclass logistic regression
void fun2() {
	//set parameters for encryption
	seal::EncryptionParameters parms;
	parms.set_poly_modulus("1x^4096 + 1");
	parms.set_coeff_modulus(
			seal::ChooserEvaluator::default_parameter_options().at(4096));
	parms.set_plain_modulus(1 << 8);
	parms.set_decomposition_bit_count(58);
	parms.validate();

	// initialize random nubmer generator
	random_device rd;
	mt19937 rng(rd());

	// logistic regression
	//read in data for classification
	arma::mat train_dat_reg(5, 101);
	arma::vec resp_train_reg;
	arma::mat test_dat_reg;
	arma::vec resp_test_reg;

	data::Load("iris.csv", test_dat_reg);

	for (int i = 0; i < 101; i++) {
		uniform_int_distribution<int> uni(1, 149 - i);
		int random_integer = uni(rng);
		for (int j = 0; j < 5; j++) {
			double tmp = test_dat_reg(j, random_integer);
			train_dat_reg(j, i) = tmp;
		}
		test_dat_reg.shed_col(random_integer);
	}
	arma::mat tmp_reg = train_dat_reg.t();
	arma::mat tmp2_reg = test_dat_reg.t();
	resp_train_reg = tmp_reg.col(tmp_reg.n_cols - 1);
	resp_test_reg = tmp2_reg.col(tmp_reg.n_cols - 1);

	train_dat_reg.shed_row(train_dat_reg.n_rows - 1);
	test_dat_reg.shed_row(test_dat_reg.n_rows - 1);

	data::Save("traintar.csv", resp_train_reg);
	data::Save("testtr.csv", resp_test_reg);
	data::Save("traindat.csv", train_dat_reg);
	data::Save("test.csv", test_dat_reg);

	//number of columns and rows for training and test set
	int train_col_reg = static_cast<int>(train_dat_reg.n_cols);
	int train_row_reg = static_cast<int>(train_dat_reg.n_rows);
	int test_col_reg = static_cast<int>(test_dat_reg.n_cols);
	int test_row_reg = static_cast<int>(test_dat_reg.n_rows);

	//Generate Keys
	seal::KeyGenerator generator(parms);
	generator.generate();
	const seal::BigPolyArray p_key = generator.public_key();
	seal::Ciphertext public_key = seal::Ciphertext(p_key);
	const seal::BigPoly s_key = generator.secret_key();
	seal::Plaintext secret_key = seal::Plaintext(s_key);
	generator.generate_evaluation_keys(test_col_reg);
	const seal::EvaluationKeys evkey = generator.evaluation_keys();
	seal::FractionalEncoder frencoder(parms.plain_modulus(),
			parms.poly_modulus(), 64, 32, 3);
	seal::IntegerEncoder encoder(parms.plain_modulus());
	seal::Evaluator evaluate(parms, evkey);
	//seal::Evaluator evaluate(par);
	seal::Encryptor encryptor(parms, p_key);
	seal::Decryptor decryptor(parms, s_key);

	//#columns, dataset, theta, Evaluator

	seal::Ciphertext ** encoded_train_reg = new seal::Ciphertext*[train_col_reg];
	seal::Ciphertext ** encoded_test_reg = new seal::Ciphertext*[test_col_reg];
	seal::Ciphertext* test_resp_reg = new seal::Ciphertext[test_col_reg];
	seal::Ciphertext* train_resp_reg_0 = new seal::Ciphertext[train_col_reg];
	seal::Ciphertext* train_resp_reg_1 = new seal::Ciphertext[train_col_reg];
	seal::Ciphertext* train_resp_reg_2 = new seal::Ciphertext[train_col_reg];
	auto start = chrono::steady_clock::now();
	seal::Ciphertext one_encrypt = encrypt_frac(1.0, encryptor, frencoder);
	seal::Ciphertext zero_encrypt = encrypt_frac(0.0, encryptor, frencoder);
	//	encrypt targets y (1 x columns)
	for (int i = 0; i < train_col_reg; i++) {
		// Ciphertext t = encrypt_frac(resp_train_reg(i), encryptor, frencoder);
		if (resp_train_reg(i) == 0) {
			train_resp_reg_0[i] = one_encrypt;
			train_resp_reg_1[i] = zero_encrypt;
			train_resp_reg_2[i] = zero_encrypt;
		} else if (resp_train_reg(i) == 1) {
			train_resp_reg_0[i] = zero_encrypt;
			train_resp_reg_1[i] = one_encrypt;
			train_resp_reg_2[i] = zero_encrypt;

		} else {
			train_resp_reg_0[i] = zero_encrypt;
			train_resp_reg_1[i] = zero_encrypt;
			train_resp_reg_2[i] = one_encrypt;

		}
	}
	auto end = chrono::steady_clock::now();
	auto diffr = end - start;
	cout << "encrypt train targets "

	<< chrono::duration<double, ratio<1>>(diffr).count() << " s" << endl;

	start = chrono::steady_clock::now();
	for (int i = 0; i < test_col_reg; i++) {
		test_resp_reg[i] = encrypt_frac(resp_test_reg(i), encryptor, frencoder);
	}
	end = chrono::steady_clock::now();
	diffr = end - start;
	cout << "encrypt test targets "

	<< chrono::duration<double, ratio<1>>(diffr).count() << " s" << endl;

	start = chrono::steady_clock::now();
	for (int i = 0; i < train_col_reg; i++) {
		encoded_train_reg[i] = new seal::Ciphertext[train_row_reg + 1];
		encoded_train_reg[i][0] = one_encrypt;
		for (int j = 1; j <= train_row_reg; j++) {
			encoded_train_reg[i][j] = encrypt_frac(train_dat_reg(j - 1, i),
					encryptor, frencoder);
		}
	}
	end = chrono::steady_clock::now();
	diffr = end - start;
	cout << "encrypt train data "

	<< chrono::duration<double, ratio<1>>(diffr).count() << " s" << endl;

	start = chrono::steady_clock::now();
	//	seal::Ciphertext one_encrypt = encrypt_frac(1.0, encryptor, frencoder);
	for (int i = 0; i < test_col_reg; i++) {
		encoded_test_reg[i] = new seal::Ciphertext[test_row_reg + 1];
		encoded_test_reg[i][0] = one_encrypt;
		for (int j = 1; j <= test_row_reg; j++) {
			encoded_test_reg[i][j] = encrypt_frac(test_dat_reg(j - 1, i),
					encryptor, frencoder);
		}
	}
	end = chrono::steady_clock::now();
	diffr = end - start;
	cout << "encrypt test data "

	<< chrono::duration<double, ratio<1>>(diffr).count() << " s" << endl;

	// training rate alpha
	double alph_reg = 0.05;
	// tr = lambda/2*col, lambda security parameter?
	double lambda_reg = 0.4;
	seal::Plaintext fr = frencoder.encode(1 / (1.0 * train_col_reg));
	seal::Plaintext a0 = frencoder.encode(-0.714761);
	seal::Plaintext a1 = frencoder.encode(-0.5);
	seal::Plaintext a2 = frencoder.encode(-0.0976419);
	seal::Plaintext minus_one_reg = frencoder.encode(-1.0);
	seal::Plaintext tr = frencoder.encode(lambda_reg / (2.0 * train_col_reg));
	seal::Plaintext alpha_reg = frencoder.encode(alph_reg);
	LogisticRegression lr; // = LogisticRegression::LogisticRegression();

	// initialize theta for regression and classification with 1 and encrypt it (row x 1)

	seal::Ciphertext **theta_reg = new seal::Ciphertext*[test_row_reg + 1];

//	for (int i = 0; i <= test_row_reg; i++) {
//		theta_reg[i] = new seal::Ciphertext[1];
//		theta_reg[i][0] = one_encrypt;
//
//	}

	for (int i = 0; i <= test_row_reg; i++) {
		theta_reg[i] = new seal::Ciphertext[3];
		for (int j = 0; j < 3; j++) {
			theta_reg[i][j] = one_encrypt;
		}

	}
	// train the classification model
	start = chrono::steady_clock::now();
	seal::Ciphertext **weights_reg = new seal::Ciphertext*[test_row_reg + 1];
	weights_reg = lr.train_multiclass(15, train_col_reg, train_row_reg,
			encoded_train_reg, theta_reg, train_resp_reg_0, train_resp_reg_1,
			train_resp_reg_2, evaluate, fr, a0, a1, a2, minus_one_reg, tr,
			alpha_reg, 3);
	end = chrono::steady_clock::now();
	diffr = end - start;
	cout << "train" << chrono::duration<double, ratio<60>>(diffr).count()
			<< " min" << endl;
	for (int i = 0; i <= train_row_reg; i++) {
		for (int j = 0; j < 3; j++) {
			cout << decrypt_frac(weights_reg[i][j], decryptor, frencoder)
					<< endl;
		}
	}
	//delete initialized theta
	//	delete[] theta_reg[0];
	//	delete[] theta_reg;

	start = chrono::steady_clock::now();
	//make multiclass predictions

	int k = 3;
	seal::Ciphertext** predictions_reg = new seal::Ciphertext*[test_col_reg];
	predictions_reg = lr.predict_multiclass(k, test_col_reg, test_row_reg,
			encoded_test_reg, weights_reg, a0, a1, a2, evaluate);
	//	seal::Ciphertext** predictions_reg = new seal::Ciphertext*[test_col_reg];
	//	predictions_reg = lr.predict_multiclass(k, test_col_reg, test_row_reg,
	//			encoded_test_reg, theta_reg, a0, a1, a2, evaluate);
	end = chrono::steady_clock::now();
	diffr = end - start;
	cout << "prediction " << chrono::duration<double, ratio<60>>(diffr).count()
			<< " min" << endl;

	double ** predictions_reg_encrypted = new double*[test_col_reg];
	int * class_prediction = new int[test_col_reg];
	start = chrono::steady_clock::now();
	for (int j = 0; j < test_col_reg; j++) {
		predictions_reg_encrypted[j] = new double[k];
		for (int i = 0; i < k; i++) {
			predictions_reg_encrypted[j][i] = decrypt_frac(
					predictions_reg[j][i], decryptor, frencoder);

		}
	}
	// make multiclass predictions: take maximum of all the classifiers (One-vs-all)
	for (int i = 0; i < test_col_reg; i++) {
		double max = predictions_reg_encrypted[i][0];
//		cout << " 0 " << max << endl;
		int cl = 0;
		for (int j = 1; j < k; j++) {
			if (predictions_reg_encrypted[i][j] > max) {
				//		cout << j << " " << max << endl;
				max = predictions_reg_encrypted[i][j];
				cl = j;
				//		cout << "pred class " << cl << endl;
			}
			if (j == k - 1) {
				class_prediction[i] = cl;
				//	cout << "end pred class " << cl << endl;
			}
		}
	}

	end = chrono::steady_clock::now();
	diffr = end - start;
	cout << "decrypt test data "

	<< chrono::duration<double, ratio<1>>(diffr).count() << " s" << endl;

//	// true positive: # of correctly recognized observations for class i
	int *tp = new int[k];
	//false positive: # of observations that were incorrectly assigned to the class C i
	int *fp = new int[k];
	// false negative: the number of observations that were not recognized as belonging to the class i
	int *fn = new int[k];
	// true negative: # of correctly recognized observations that do not belong to the class i
	int *tn = new int[k];

	for (int i = 0; i < test_col_reg; i++) {
		double max = predictions_reg_encrypted[i][0];
//		cout << " 0 " << max << endl;
		int cl = 0;
		for (int j = 1; j < k; j++) {
			if (predictions_reg_encrypted[i][j] > max) {
				//		cout << j << " " << max << endl;
				max = predictions_reg_encrypted[i][j];
				cl = j;
				//		cout << "pred class " << cl << endl;
			}
			if (j == k - 1) {
				class_prediction[i] = cl;
				//	cout << "end pred class " << cl << endl;
			}
		}
	}
	for (int i = 0; i < test_col_reg; i++) {
		for (int j = 0; j < k; j++) {
			double max = predictions_reg_encrypted[i][j];
//		cout << " 0 " << max << endl;
			if (max < 0.5) {
				if (resp_test_reg(i) == 0) {
					if (j == 0) {
						fn[j]++;
						fp[1]++;
						fp[2]++;
					} else {
						tp[1]++;
						tp[2]++;
						tn[0]++;
					}
				} else if (resp_test_reg(i) == 1) {
					if (j == 1) {
						fn[j]++;
						fp[0]++;
						fp[2]++;
					} else {
						tp[0]++;
						tp[2]++;
						tn[1]++;
					}
				} else {
					if (j == 2) {
						fn[j]++;
						fp[1]++;
						fp[0]++;
					} else {
						tp[1]++;
						tp[2]++;
						tn[0]++;
					}
				}
			} else {
				if (resp_test_reg(i) == 0) {
					if (j == 0) {
						tp[j]++;
						tn[1]++;
						tn[2]++;
					} else {
						fn[0]++;
						fp[1]++;
						fp[2]++;
					}
				} else if (resp_test_reg(i) == 1) {
					if (j == 1) {
						tp[j]++;
						tn[0]++;
						tn[2]++;
					} else {
						fn[1]++;
						fp[0]++;
						fp[2]++;
					}
				} else {
					if (j == 2) {
						tp[j]++;
						tn[0]++;
						tn[1]++;
					} else {
						fn[2]++;
						fp[0]++;
						fp[1]++;
					}
				}

			}
		}
	}

	/*for (int j = 0; j < k; j++) {
	 for (int i = 0; i < test_col_reg; i++) {
	 if (predictions_reg_encrypted[i][j] == 0) {
	 if (resp_test_reg(i) == 0) {
	 tp[0] += 1;
	 }
	 if (resp_test_reg(i) == 1) {
	 fp[0] += 1;
	 fn[1] += 1;
	 }
	 if (resp_test_reg(i) == 2) {
	 fp[0] += 1;
	 fn[2] += 1;
	 }

	 }
	 if (predictions_reg_encrypted[i][j] == 1) {

	 if (resp_test_reg(i) == 1) {
	 tp[1] += 1;
	 }
	 if (resp_test_reg(i) == 0) {
	 fp[1] += 1;
	 fn[0] += 1;
	 }
	 if (resp_test_reg(i) == 2) {
	 fp[1] += 1;
	 fn[2] += 1;
	 }
	 }
	 if (predictions_reg_encrypted[i][j] == 2) {
	 if (resp_test_reg(i) == 2) {
	 tp[2] += 1;
	 }
	 if (resp_test_reg(i) == 1) {
	 fp[2] += 1;
	 fn[1] += 1;
	 }
	 if (resp_test_reg(i) == 0) {
	 fp[2] += 1;
	 fn[0] += 1;
	 }
	 }
	 }
	 tn[0] = tp[1] + tp[2];
	 tn[1] = tp[0] + tp[2];
	 tn[2] = tp[1] + tp[0];
	 } */
	int* sum = new int[k];
	double* acc = new double[k];
	for (int i = 0; i < k; i++) {
		sum[i] = tp[i] + tn[i];
		acc[i] = (1.0 * sum[i]) / (1.0 * test_col_reg);
		cout << "accuracy of classifier " << i << " is " << acc[i] << endl;
	}

}

//uncomment when running linear regression
void fun3() {
	// set parameters for encryption
	seal::EncryptionParameters parms;
	parms.set_poly_modulus("1x^4096 + 1");
	parms.set_coeff_modulus(
			seal::ChooserEvaluator::default_parameter_options().at(4096));
	parms.set_plain_modulus(1 << 8);
	parms.set_decomposition_bit_count(58);
	parms.validate();

	//Generate Keys for encryption
	seal::KeyGenerator generator(parms);
	generator.generate();
	const seal::BigPolyArray p_key = generator.public_key();
	seal::Ciphertext public_key = seal::Ciphertext(p_key);
	const seal::BigPoly s_key = generator.secret_key();
	seal::Plaintext secret_key = seal::Plaintext(s_key);

	// initialize random nubmer generator
	random_device rd;
	mt19937 rng(rd());

	//read in data for classification
	arma::mat train_dat(9, 601);
//	arma::mat train_dat(9, 101);
	arma::vec resp_train;
	arma::mat test_dat;
	arma::vec resp_test;
	data::Load("abalone.csv", test_dat);

	int i = 0;
	while (i < 601) {
//	while (i < 101) {
		uniform_int_distribution<int> uni(0, 4176 - i);
		int random_integer = uni(rng);

		if (random_integer != 8) {
			for (int j = 0; j < 9; j++) {
				double tmp = test_dat(j, random_integer);
				//cout << "fine" << endl;
				train_dat(j, i) = tmp;
			}

			test_dat.shed_col(random_integer);
			i++;
		}
	}

//	for (int i = 0; i < 1301; i++) {
//		uniform_int_distribution<int> uni(0, 4177 - i);
//		int random_integer = uni(rng);
//		cout << i << " and " << random_integer << endl;
//		for (int j = 0; j < 9; j++) {
//			double tmp = test_dat(j, random_integer);
//			//		cout << "fine" << endl;
//			train_dat(j, i) = tmp;
//		}
//		//	cout << "before shed" << endl;
//		test_dat.shed_col(random_integer);
//	}
	arma::mat tmp = train_dat.t();

	arma::mat tmp2 = test_dat.t();
	resp_train = tmp.col(8);	// label of train data
	resp_test = tmp2.col(8);
	train_dat.shed_row(train_dat.n_rows - 1);
	train_dat.shed_row(0);
	test_dat.shed_row(0);
	test_dat.shed_row(test_dat.n_rows - 1);
	data::Save("test_reg.csv", test_dat);
	data::Save("train_reg.csv", train_dat);
	data::Save("test_reg_tr.csv", resp_test);
	data::Save("train_reg_tr.csv", resp_test);

	//number of columns and rows for training and test set
	int train_col = static_cast<int>(train_dat.n_cols);
	int train_row = static_cast<int>(train_dat.n_rows);
	int test_col = static_cast<int>(test_dat.n_cols);
	int test_row = static_cast<int>(test_dat.n_rows);

	// generate keys in order to relinearize ciphertexts
	generator.generate_evaluation_keys(25);
	const seal::EvaluationKeys evkey = generator.evaluation_keys();
	seal::FractionalEncoder frencoder(parms.plain_modulus(),
			parms.poly_modulus(), 64, 32, 3);
	seal::IntegerEncoder encoder(parms.plain_modulus());
	seal::Evaluator evaluate(parms, evkey);
	//seal::Evaluator evaluate(par);
	seal::Encryptor encryptor(parms, p_key);
	seal::Decryptor decryptor(parms, s_key);

	// encoding and encrpypting the polynomials
	seal::Ciphertext **encoded_train = new seal::Ciphertext *[train_col];

	//const seal::EncryptionParameters par = parms;

	// encrypt the given data

//  encrypt train data
	auto start = chrono::steady_clock::now();

	seal::Ciphertext one_encrypt = encrypt_frac(1.0, encryptor, frencoder);

	for (int i = 0; i < train_col; i++) {
		encoded_train[i] = new seal::Ciphertext[train_row + 1];
		encoded_train[i][0] = one_encrypt;
		for (int j = 1; j <= train_row; j++) {
			encoded_train[i][j] = encrypt_frac(train_dat(j - 1, i), encryptor,
					frencoder);
		}
	}
	auto end = chrono::steady_clock::now();
	auto diffr = end - start;
	cout << "encrypt train data "
			<< chrono::duration<double, ratio<1>>(diffr).count() << " s"
			<< endl;
	seal::Ciphertext *train_resp = new seal::Ciphertext[train_col];
	start = chrono::steady_clock::now();

	//	encrypt train labels y (1 x columns)
	for (int i = 0; i < train_col; i++) {

		train_resp[i] = encrypt_frac(resp_train(i), encryptor, frencoder);

	}

	end = chrono::steady_clock::now();
	diffr = end - start;
	cout << "encrypt train targets "
			<< chrono::duration<double, ratio<1>>(diffr).count() << " s"
			<< endl;

	// encrypt test labels

	seal::Ciphertext *test_resp = new seal::Ciphertext[251];
	start = chrono::steady_clock::now();
	for (int i = 0; i < 251; i++) {
		test_resp[i] = encrypt_frac(resp_test(i), encryptor, frencoder);
	}
	end = chrono::steady_clock::now();
	diffr = end - start;
	cout << "encrypt test targets "
			<< chrono::duration<double, ratio<60>>(diffr).count() << " min"
			<< endl;

	seal::Ciphertext **encoded_test = new seal::Ciphertext *[test_col];

	start = chrono::steady_clock::now();
	for (int i = 0; i < 251; i++) {
		encoded_test[i] = new seal::Ciphertext[test_row + 1];
		encoded_test[i][0] = one_encrypt;
		for (int j = 1; j <= test_row; j++) {
			encoded_test[i][j] = encrypt_frac(test_dat(j - 1, i), encryptor,
					frencoder);
		}
	}
	end = chrono::steady_clock::now();
	diffr = end - start;
	cout << "encrypt test data "
			<< chrono::duration<double, ratio<60>>(diffr).count() << " min"
			<< endl;

	//learning rate alpha = 0.32
	double alph = 0.032 / train_col;
	seal::Plaintext alpha = frencoder.encode(alph);
	//	constant (1.0 / (2 * n_col)
	seal::Plaintext text = frencoder.encode((1.0 / (2.0 * train_col)));
	// seal::Ciphertext one_encrypt = encrypt_frac(1.0, encryptor, frencoder);
	double lambda = 1.5 / (1.0 * train_col);
	seal::Plaintext lambda_div = frencoder.encode(lambda);
	// initialize the weights
	seal::Ciphertext **theta = new seal::Ciphertext *[train_row + 1];
	for (int i = 0; i <= train_row; i++) {
		theta[i] = new seal::Ciphertext[1];
		theta[i][0] = one_encrypt;

	}
// for (int i = 0; i <= 7; i++) {
// theta[i] = new seal::Ciphertext[1];
// //cout << i << endl;
// }
// theta[0][0] = encrypt_frac(1.913, encryptor, frencoder);
// //	cout << "theta fine" << endl;
// theta[1][0] = encrypt_frac(10.813, encryptor, frencoder);
// //	cout << "theta fine" << endl;
// theta[2][0] = encrypt_frac(4.175, encryptor, frencoder);
// //cout << "theta fine" << endl;
// theta[3][0] = encrypt_frac(-1.36, encryptor, frencoder);
// //	cout << "theta fine" << endl;
// theta[4][0] = encrypt_frac(8.053, encryptor, frencoder);
// //	cout << "theta fine" << endl;
// theta[5][0] = encrypt_frac(-7.258, encryptor, frencoder);
// //	cout << "theta fine" << endl;
// theta[6][0] = encrypt_frac(-4.66, encryptor, frencoder);
// //cout << "theta fine" << endl;
// theta[7][0] = encrypt_frac(4.58, encryptor, frencoder);
// cout << "theta fine" << endl;
	bool ridge = false;

	OwnLinearRegression linreg;
	start = chrono::steady_clock::now();
	// train the linear regression model
	seal::Ciphertext **weights = new seal::Ciphertext*[train_row + 1];
	weights = linreg.train(alpha, 3, train_col, encoded_train, train_resp,
			theta, train_row, text, evaluate, ridge, lambda_div);
	end = chrono::steady_clock::now();
	diffr = end - start;
	cout << "train " << chrono::duration<double, ratio<60>>(diffr).count()
			<< " min" << endl;
	//	delete pre - initialized theta
// 	delete[] theta[0];
// 	delete[] theta;
	for (int i = 0; i <= train_row; i++) {
		cout << decrypt_frac(theta[i][0], decryptor, frencoder) << endl;
	}
	//make predicitons on test data set
	start = chrono::steady_clock::now();
	seal::Ciphertext *pred = linreg.predict(test_row, 251, encoded_test,
			weights, evaluate);
	end = chrono::steady_clock::now();
	diffr = end - start;
	cout << "pred " << chrono::duration<double, ratio<60>>(diffr).count()
			<< " min" << endl;
	double *dec = new double[test_col];
	// later: compare prediction on unencrypted and encrypted data

	start = chrono::steady_clock::now();
	double MSE = 0;
	for (int i = 0; i < 251; i++) {
		double d = decrypt_frac((*(pred + i)), decryptor, frencoder);
		dec[i] = d;
		cout << dec[i] << " actual " << resp_test(i) << endl;
		double tmp = (dec[i] - resp_test(i));
		MSE += pow(tmp, 2);
	}
	cout << " Means squared error " << MSE / (100 * 1.0) << endl;
	end = chrono::steady_clock::now();
	diffr = end - start;
	cout << "decryption " << chrono::duration<double, ratio<60>>(diffr).count()
			<< " min" << endl;
}

int main() {
	// binary logistic regression
	//fun1();
//multiclass logistic regression
	// fun2();
	//linear regression
	fun3();

	return 0;
}

