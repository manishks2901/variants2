// MongoDB initialization script
// This script runs when the container starts for the first time

// Switch to the video_variants database
db = db.getSiblingDB('video_variants');

// Create collections with indexes for better performance
db.createCollection('jobs');
db.createCollection('users');

// Create indexes for jobs collection
db.jobs.createIndex({ "job_id": 1 }, { unique: true });
db.jobs.createIndex({ "status": 1 });
db.jobs.createIndex({ "created_at": 1 });
db.jobs.createIndex({ "user_id": 1 });

// Create indexes for users collection (if needed)
db.users.createIndex({ "user_id": 1 }, { unique: true });

print('MongoDB initialization completed');
