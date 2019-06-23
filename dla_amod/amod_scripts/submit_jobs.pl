#!/home/utils/perl-5.8.8/bin/perl

use Getopt::Long;
use Data::Dumper;
use List::Util (sum);

## use lib '/home/nv/utils/nvperl/1.8.1/lib';
use lib '/home/nv/utils/nvbatch/1.8.11/lib/perl5';
use NV::Batch::INC;
use NV::Batch;
use File::Spec::Functions;

my $model_def   = "";
my $pre_trained = "";
my $out_dir     = "";
my $histdir     = "";
my $hist_mode   = -1;
my $skip        = 0;
my $interval    = 1;
my $batch       = 50;
my $maxWaitTime = 10;       # Minutes
my $db          = 'DLA_AMOD_SIMU';

GetOptions (
	    "model_def=s"   => \$model_def,         # string
	    "pre_trained=s" => \$pre_trained,       # string
	    "out_dir=s"     => \$out_dir,           # string
	    "histdir=s"     => \$histdir,           # string
	    "hist_mode=i"   => \$hist_mode,         # integer
	    "skip=i"        => \$skip,              # integer
	    "interval=i"    => \$interval,          # integer
	    "batch=i"       => \$batch,             # integer
	    "maxtime=i"     => \$maxWaitTime,       # integer
	    "db=s"          => \$db,                # string
	   ) or die ("Error in command line arguments\n");

#### CONFIG START ###
my $total_images = 50000;
my $total_process= 500;
my $num_db       = 10;
#### CONFIG END ###

my $workspace   = '/home/scratch.yilinz_t19x/git/dla_amod';
my $batch_size   = $batch;
my $images_per_process = $total_images/$total_process;
my $iter = $images_per_process/($batch_size*$interval);
my $process_per_db = $total_process/$num_db;
my $tree_dir    = './batch_logs';


my $my_tuning_jobs;
if ( $histdir eq "" ) {
    $my_tuning_jobs = $workspace.'/build/tools/caffe test -model '.$model_def.' -iterations '.$iter.' -weights '.$pre_trained.' -histfiles 0 -batch '.$batch_size.' -interval '.$interval.' ';
} else {
    if ($hist_mode eq 0) {
        # Collect data range only
        $my_tuning_jobs = $workspace.'/build/tools/caffe test -model '.$model_def.' -iterations '.$iter.' -weights '.$pre_trained.' -histfiles -1 -histdir '.$histdir.' -batch '.$batch_size.' -interval '.$interval.' ';
    } elsif ($hist_mode eq 1) {
        # Collect histogram
        $my_tuning_jobs = $workspace.'/build/tools/caffe test -model '.$model_def.' -iterations '.$iter.' -weights '.$pre_trained.' -histfiles '.$total_process.' -histdir '.$histdir.' -batch '.$batch_size.' -interval '.$interval.' ';
    }
}

my @task_list = ($my_tuning_jobs);
foreach $task(@task_list)
{
    my @tests = map {
    NV::Batch->newTest(
    name    => 'test'.$_,
    farmId  => $_,
    command => $task.' -skip '.($skip+$_*$images_per_process).' -process '.$_.' -dbfile '.$workspace.'/../db/val_db'.(int($_/$process_per_db)),
    dir     => catfile($out_dir,'batch_'.$_),
    )
    } 0..$total_process-1;

    my $batch = NV::Batch->new(
        dir      => $tree_dir,
        testList => \@tests,
    );

    $batch->generate();
    local $SIG{ALRM} = sub { alarm 0; print "timout reached, killing batch\n"; $batch->kill(); };
    alarm $maxWaitTime*60;

    $batch->run(
        farm    => 1,
        farmSessionTemplate => {
            databaseID => $db,            # <-- AND HERE
            resourcePredictorOptions => {
                retryTimeFactor     => 2,
                retryTimeIncrement  => 0,
                retryMemFactor      => 1.5,
                retryMemIncrement   => 400,
            },
        },
        testJobTemplate => {
            # It seems only 5 jobs can running in parallel if *_pri_* queue is selected
            queue => 'o_cpu_8G_1H,o_cpu_8G_4H,o_regress_cpu_8G',
            maxRunTime  => $maxWaitTime*60,
            submitOptions => [ '-m rel68' ],
        },
        rerunJobTemplate => {
            queue => 'o_cpu_16G',
            submitOptions => [ '-m rel68' ],
        },
        rerunPatterns => ['^ERROR_TERM_*'], # only rerun when killed by farm 
    );

    print "Task finished\n";
    alarm 0;
}
