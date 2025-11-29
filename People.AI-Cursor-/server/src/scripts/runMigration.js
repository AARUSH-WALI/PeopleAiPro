/**
 * Script to run the job_id migration
 * Usage: node src/scripts/runMigration.js
 */

import { supabase } from '../config/supabase.js';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function runMigration() {
    try {
        console.log('Reading migration file...');
        const migrationPath = path.join(__dirname, '../../migrations/010_add_job_id_to_resumes.sql');
        const migrationSQL = fs.readFileSync(migrationPath, 'utf8');

        console.log('Executing migration...');
        const { data, error } = await supabase.rpc('exec_sql', { sql: migrationSQL });

        if (error) {
            // If RPC doesn't work, try executing directly via query
            console.log('RPC method not available, trying direct execution...');

            // Split the SQL into individual statements
            const statements = migrationSQL
                .split(';')
                .map(s => s.trim())
                .filter(s => s.length > 0 && !s.startsWith('--'));

            for (const statement of statements) {
                if (statement.trim()) {
                    console.log(`Executing: ${statement.substring(0, 50)}...`);
                    const { error: execError } = await supabase.from('_migrations').select('*').limit(0);

                    // Actually, Supabase client doesn't support raw SQL execution directly
                    // We need to use the REST API or psql
                    console.error('Direct SQL execution not supported via Supabase client.');
                    console.error('Please run the migration using one of these methods:');
                    console.error('1. Supabase Dashboard SQL Editor');
                    console.error('2. psql command line');
                    console.error('3. See the SQL in: server/migrations/010_add_job_id_to_resumes.sql');
                    return;
                }
            }
        }

        console.log('Migration executed successfully!');
    } catch (error) {
        console.error('Error running migration:', error);
        console.error('\nPlease run the migration manually using one of these methods:');
        console.error('\n1. Supabase Dashboard:');
        console.error('   - Go to SQL Editor');
        console.error('   - Paste the SQL from: server/migrations/010_add_job_id_to_resumes.sql');
        console.error('   - Click Run');
        console.error('\n2. Using psql:');
        console.error('   psql -h <your-db-host> -U postgres -d postgres -f server/migrations/010_add_job_id_to_resumes.sql');
    }
}

runMigration();


