/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 10;
	default_random_engine gen;
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for (int i = 0; i < num_particles; ++i) {
        Particle particle{i, dist_x(gen), dist_y(gen), dist_theta(gen), 1};
        particles.push_back(particle);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    default_random_engine gen;
    normal_distribution<double> dist_x(0, std_pos[0]);
    normal_distribution<double> dist_y(0, std_pos[1]);
    normal_distribution<double> dist_theta(0, std_pos[2]);
    for (Particle &particle: particles) {
        if (fabs(yaw_rate) > 0.001) {
            particle.x += velocity / yaw_rate * (sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta));
            particle.y += velocity / yaw_rate * (-cos(particle.theta + yaw_rate * delta_t) + cos(particle.theta));
            particle.theta += yaw_rate * delta_t;
        } else {
            particle.x += velocity * cos(particle.theta) * delta_t;
            particle.y += velocity * sin(particle.theta) * delta_t;
            particle.theta += 0;
        }
        particle.x += dist_x(gen);
        particle.y += dist_y(gen);
        particle.theta += dist_theta(gen);
    }

}

double squared(double x) {
    return x * x;
}

double computeGaussianProbability(double x, double y, double mu_x, double mu_y, double std_x, double std_y) {
    return exp(-(squared(x - mu_x) / (2 * squared(std_x)) + squared(y - mu_y) / (2 * squared(std_y)))) /
           (2 * M_PI * std_x * std_y);
}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
    // Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html

    // Identify the landmark each observation is associated with
    for (Particle &particle: particles) {
        std::vector<double> sense_x_world;
        std::vector<double> sense_y_world;
        std::vector<int> associations;
        for (const LandmarkObs &obs: observations) {
            // Transform observation from car coordinate system to map coordinate system
            double x_world = particle.x + obs.x * cos(-particle.theta) + obs.y * sin(-particle.theta);
            double y_world = particle.y + obs.x * sin(particle.theta) + obs.y * cos(particle.theta);

            // Find the closest Landmark
            int max_i = 0;
            double min_dist = sensor_range;
            for (Map::single_landmark_s landmark: map_landmarks.landmark_list) {
                double distance = dist((double)landmark.x_f, (double)landmark.y_f, x_world, y_world);
                if (distance < min_dist) {
                    max_i = landmark.id_i;
                    min_dist = distance;
                }
            }

            // Use only observations with landmark associated
            if (max_i > 0) {
                double dd = dist(8.7638, 7.5647, x_world, y_world);
                sense_x_world.push_back(x_world);
                sense_y_world.push_back(y_world);
                associations.push_back(max_i);
            }
        }

        particle = SetAssociations(particle, associations, sense_x_world, sense_y_world);

        // Calculate weight based on Gaussian distribution
        particle.weight = 1.;
        for (int i = 0; i < particle.associations.size(); ++i) {
            int landmark_i = particle.associations[i];
            double x_sense = particle.sense_x[i];
            double y_sense = particle.sense_y[i];

            double x_landmark = sensor_range;
            double y_landmark = sensor_range;
            for (const Map::single_landmark_s &landmark: map_landmarks.landmark_list) {
                if (landmark.id_i == landmark_i){
                    x_landmark = landmark.x_f;
                    y_landmark = landmark.y_f;
                    break;
                }
            }
            double prob = computeGaussianProbability(x_sense, y_sense, x_landmark, y_landmark, std_landmark[0], std_landmark[1]);
            particle.weight *= prob;
        }
    }
}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    default_random_engine gen;

    std::vector<double> weights;
    for (const auto &particle: particles) {
        weights.push_back(particle.weight);
    }
    std::discrete_distribution<> d(weights.begin(), weights.end());
    std::vector<Particle> new_particles;
    for (int i=0; i < num_particles; ++i) {
        new_particles.push_back(particles[d(gen)]);
    }
    particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
