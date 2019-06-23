#!/home/utils/perl-5.8.8/bin/perl

use Getopt::Long;
use Data::Dumper;
use List::Util (sum);

## use lib '/home/nv/utils/nvperl/1.8.1/lib';
use lib '/home/nv/utils/nvbatch/1.8.11/lib/perl5';
use NV::Batch::INC;
use NV::Batch;
use File::Spec::Functions;

my $job_file    = "";
my $out_dir     = "";
my $maxWaitTime = 30;       # Minutes

GetOptions (
	    "job_file=s"   => \$job_file,         # 
	    "out_dir=s"     => \$out_dir,           # string
	   ) or die ("Error in command line arguments\n");

my $tree_dir    = './batch_logs';
my @task_list = ();
if (open(my $fh, '<:encoding(UTF-8)', $job_file)) {
    while (my $row = <$fh>) {
        chomp $row;
        push @task_list, $row;
    } 
};

my @tests = ();
my $test_id = 0;
foreach $task (@task_list) {
    push @tests, NV::Batch->newTest(
        name    => 'test'.$test_id,
        command => $task,
        dir     => catfile($out_dir,'batch_'.$test_id),
        );
    $test_id += 1;
};

if ($test_id > 0) {
    my $batch = NV::Batch->new(
        dir      => $tree_dir,
        testList => \@tests,
    );
    
    $batch->generate();
    local $SIG{ALRM} = sub { alarm 0; print "timout reached, killing batch\n"; $batch->kill(); };
    alarm $maxWaitTime*60;
    
    $batch->run(
        farmSessionTemplate => {
            databaseID => 'DLA_AMOD_SIMU',            # <-- AND HERE
            #resourcePredictorOptions => {
            #    retryTimeFactor     => 2,
            #    retryTimeIncrement  => 0,
            #    retryMemFactor      => 1.5,
            #    retryMemIncrement   => 400,
            #},
        },
        testJobTemplate => {
            # It seems only 5 jobs can running in parallel if *_pri_* queue is selected
            #queue => 'o_pri_cpu_8G,o_cpu_8G_4H,o_cpu_16G,o_cpu_32G_24H',
            queue => 'o_cpu_8G_1H,o_cpu_16G,o_cpu_32G_24H',
            #queue => 'o_cpu_8G_12H,o_cpu_16G,o_cpu_32G_24H',
            submitOptions => [ '-m rel68' ],
        },
        #rerunJobTemplate => {
        #    queue => 'o_cpu_16G',
        #    submitOptions => [ '-m rel68' ],
        #},
        #rerunPatterns => ['^ERROR_TERM_*'], # only rerun when killed by farm 
    );
    alarm 0;
}

print "Task finished\n";
