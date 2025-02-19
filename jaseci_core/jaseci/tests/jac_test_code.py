ll_proto = \
    """
    node life {
        has anchor owner;
        can infer.year_from_date;
    }

    node year {
        has anchor year;
        can infer.month_from_date;
    }

    node month {
        has anchor month;
        can infer.year_from_date;
        can infer.week_from_date;
    }

    node week {
        has anchor week;
        can infer.month_from_date;
        can infer.day_from_date, date.day_from_date;
    }

    node day: has anchor day;

    node workette {
        has  name, order, date, owner, status, snooze_till;
        has note, is_MIT, is_ritual;
    }

    edge past;

    edge parent;

    walker get_day {
        has date;
        life: take --> node::year == infer.year_from_date(date);
        year: take --> node::month == infer.month_from_date(date);
        month: take --> node::week == infer.week_from_date(date);
        week: take --> node::day == date.day_from_date(date);
        day: report here;
        report false;
    }

    walker get_latest_day {
        has before_date;
        has anchor latest_day;
        if(!before_date): before_date = std.time_now();
        if(!latest_day): latest_day = 0;

        life {
            ignore --> node::year > infer.year_from_date(before_date);
            take net.max(--> node::year);
        }
        year {
            ignore node::month > infer.month_from_date(before_date);
            take net.max(--> node::month)
            else {
                ignore here;
                take <-- node::life;
            }
        }
        month {
            ignore node::week > infer.week_from_date(before_date);
            take net.max(--> node::week)
            else {
                ignore here;
                take <-- node::year == infer.year_from_date(before_date);
            }
        }
        week {
            ignore node::day > infer.day_from_date(before_date);
            take net.max(--> node::day)
            else {
                ignore here;
                take <-- node::month == infer.month_from_date(before_date);
            }
        }
        day {
            latest_day = here;
            report here;
        }
    }

    walker get_gen_day {
        has date;
        has anchor day_node;
        if(!date): date=std.time_now();
        root: take --> node::life;
        life: take --> node::year == infer.year_from_date(date) else {
                new = spawn here --> node::year ;
                new.year = infer.year_from_date(date);
                take --> node::year == infer.year_from_date(date);
            }
        year: take --> node::month == infer.month_from_date(date) else {
                new = spawn here --> node::month;
                new.month = infer.month_from_date(date);
                take --> node::month == infer.month_from_date(date);
            }
        month: take --> node::week == infer.week_from_date(date) else {
                new = spawn here --> node::week;
                new.week = infer.week_from_date(date);
                take --> node::week == infer.week_from_date(date);
            }
        week: take --> node::day == infer.day_from_date(date) else {
                latest_day = spawn here walker::get_latest_day;
                new = spawn here --> node::day;
                new.day = infer.day_from_date(date);
                if(latest_day and infer.day_from_date(date) ==
                    infer.day_from_date(std.time_now())) {
                    spawn latest_day walker::carry_forward(parent=new);
                    take new;
                }
                elif(latest_day) {
                    take latest_day;
                }
                else: take new;
            }
        day {
            day_node = here;
            take --> node::workette;
        }
        workette {
            report here;
            take --> node::workette;
        }
    }

    walker get_sub_workettes {
        report here;
        workette: take --> node::workette;
    }

    walker carry_forward {
        has parent;
        day {
            take --> node::workette;
        }
        workette {
            if(here.status == 'done' or
            here.status == 'eliminated') {
                disengage;
            }
            new_workette = spawn here <-[past]- node::workette;
            new_workette <-[parent]- parent;
            new_workette := here;
            spawn --> node::workette
                walker::carry_forward(parent=new_workette);
        }
    }

    walker gen_rand_life {
        has num_workettes;
        root: take --> node::life;

        life {
            num_workettes = 10;
            num_days = rand.integer(2, 4);
            for i=0 to i<num_days by i+=1 {
                spawn here walker::get_gen_day(
                    date=rand.time("2019-01-01", "2019-12-31")
                );
            }
            take -->;
        }
        year, month, week { take -->; }
        day, workette {
            if(num_workettes == 0): disengage;
            gen_num = rand.integer(5, 8);
            for i=0 to i<gen_num by i+=1 {
                spawn here -[parent]-> node::workette(name=rand.sentence());
            }
            take --> ;
            num_workettes -= 1;
        }
    }

    walker init {
        has owner;
        has anchor life_node;
        take (--> node::life == owner) else {
            life_node = spawn here --> node::life;
            life_node.owner = owner;
            disengage;
        }
    }


    """

prog0 = \
    """
    node test:0 {
        has a, b, c;
        can std.log::a,b::>c with exit;
    }

    walker test {
        test {
            here.a = 43;
            here.b = 'Yeah \\n"fools"!';
            report here.b;
            if(4 > 6) { std.log("a"); }
            elif(5>6) { std.log("b"); }
            elif(6>6) { std.log("c"); }
            elif(7>6) { std.log(576); }
        }
    }

    node life:0 {
    }

    node year {
        has anchor year;

    }

    walker another {
        life {
            here.a = 43;
            here.b = 'Yeah \\n"fools"!';
            report here.b;
            if("4 > 6" == "4 > 6") { std.log("a"); }
        }
    }
    """

prog1 = \
    """
    node test:0 {
        has a, b, c;
        can std.log::a,b::>c with exit;
    }

    walker test {
        test {
            here.a = 43;
            here.b = 'Yeah \\n"fools"!';
            report here.b;
            if(4 > 6) { std.log("a"); }
            elif(5>6) { std.log("b"); }
            elif(6>6) { std.log("c"); }
            elif(7>6) { std.log(576); }
        }
    }

    node life:0 {
    }

    node year {
        has anchor year;

    }

    node month {
        has anchor month;
    }

    node week {
        has anchor week;
    }

    node day {
        has anchor day;
    }

    node workette {
        has date, owner, status, snooze_till;
        has note, is_MIT, is_ritual;
    }

    walker use_test {
        can use.enc_question, use.enc_answer, use.qa_score;
        has output;
        q = use.enc_question(["How old are you?",
                              "which animal is the best?"]);
        a = use.enc_answer(["I'm 40 years old.", "Elephants rule."]);
        output = use.qa_score(q, a);
        report output;
    }

    walker use_test_with_ctx {
        can use.enc_question, use.enc_answer, use.qa_score, use.dist_score;
        has output;
        q = use.enc_question("Who are you?");
        a = use.enc_answer("I am jason");
        output = use.qa_score(q, a);
        report output;
        a = use.enc_answer("You are jon");
        output = use.qa_score(q, a);
        report output;
        a = use.enc_answer("Who are you? You are jon");
        output = use.qa_score(q, a);
        report output;
        a = use.enc_answer("Who are you? You are jon");
        output = use.qa_score(q, a);
        report output;
        q1 = use.enc_question("Who are you?");
        q2 = use.enc_question("Who you be?");
        q3 = use.enc_question("Who I be?");
        output = use.dist_score(q1, q2);
        report output;
        output = use.dist_score(q1, q3);
        report output;
        output = use.qa_score(q2, use.enc_answer("Who are you? You are jon"));
        report output;
        output = use.qa_score(q3, use.enc_answer("Who are you? You are jon"));
        report output;
        output = use.qa_score(q2, use.enc_answer("I am jason"));
        report output;
        output = use.qa_score(q3, use.enc_answer("I am jason"));
        report output;
    }

    walker use_test_with_ctx2 {
        can use.enc_question, use.enc_answer, use.qa_score, use.dist_score;

        q1 = use.enc_question("Who are you?");
        q2 = use.enc_question("Who you be?");
        q3 = use.enc_question("Who I be?");
        report use.dist_score(q1, q2);
        report use.dist_score(q1, q3);
        report use.qa_score(q2, use.enc_answer("Who are you? You are jon"));
        report use.qa_score(q3, use.enc_answer("Who are you? You are jon"));
        report use.qa_score(q2, use.enc_answer("I am jason"));
        report use.qa_score(q3, use.enc_answer("I am jason"));
        report use.qa_score(q3, use.enc_answer("I am jason","Who I be?"));
        report use.qa_score(q3, use.enc_answer("I am jason Who I be?"));
    }

    walker use_test_single {
        can use.enc_question, use.enc_answer, use.qa_score;
        has output;
        q = use.enc_question("Who's your daddy?");
        a = use.enc_answer("I'm your father.");
        output = use.qa_score(q, a);
        report output;
    }

    walker get_day {
        has date;
        life: take infer.year_from_date(date);
        year: take infer.month_from_date(date);
        month: take infer.week_from_date(date);
        week: take infer.day_from_date(date);
        day: report --> ;
    }

    walker get_gen_day {
        has date;
        can infer.year_from_date;
        can infer.month_from_date;
        can infer.week_from_date;
        can infer.day_from_date;
        life: take --> node::year == infer.year_from_date(date) else {
                new = spawn here --> node::year;
                new.year = infer.year_from_date(date);
                take --> node::year == infer.year_from_date(date);
            }
        year: take --> node::month == infer.month_from_date(date) else {
                new = spawn here --> node::month;
                new.month = infer.month_from_date(date);
                take --> node::month == infer.month_from_date(date);
            }
        month: take --> node::week == infer.week_from_date(date) else {
                new = spawn here --> node::week;
                new.week = infer.week_from_date(date);
                take --> node::week == infer.week_from_date(date);
            }
        week: take --> node::day == infer.day_from_date(date) else {
                new = spawn here --> node::day;
                new.day = infer.day_from_date(date);
                take --> node::day == infer.day_from_date(date);
            }
        day: report --> ;
    }

    walker get_sub_workettes {
        workette: report --> node::workette;
    }

    walker get_latest_day {
        life: take year.max_outbound;
        year: take month.max_outbound;
        month: take week.max_outbound;
        week: report day.max_outbound;
    }

    walker carry_forward {
        has my_root;
        day {
            new_day = spawn here --> node::day;
            my_root = new_day;
            take day.outbound_nodes;
        }
        workette {
            if(workette.status == 'done' or
            workette.status == 'eliminated') {
                continue;
            }
            childern = workette.outbound_nodes;
            new_workette = spawn here --> node::workette;
            parent = me.spawn_history.last(-1);
            new_workette <-- parent;
            take --> node::workette;
        }
        report me.spawn_history;
        report new_day;
    }
    """

edgey = \
    """
    node test;

    edge apple;
    edge banana;

    walker init {
        root {
            a = spawn here --> node::test;
            here -[apple]-> a;
            here -[banana]-> a;
        }
    }
    """

edgey2 = \
    """
    node test;

    edge apple;
    edge banana;

    walker init {
        root {
            a = spawn here --> node::test;
            here -[apple]-> a;
            here -[banana]-> a;

            here !--> a;
        }
    }
    """

edgey2b = \
    """
    node test;

    edge apple;
    edge banana;

    walker init {
        root {
            a = spawn here --> node::test;
            b = spawn here --> node::test;
            here -[apple]-> a;
            here -[banana]-> a;

            here !--> a;
        }
    }
    """

edgey2c = \
    """
    node test;

    edge apple;
    edge banana;

    walker init {
        root {
            a = spawn here --> node::test;
            b = spawn here --> node::test;
            here -[apple]-> a;
            here -[banana]-> a;

            here !-[apple]-> a;
        }
    }
    """

edgey3 = \
    """
    node test;

    edge apple;
    edge banana;

    walker init {
        root {
            a = spawn here --> node::test;
            b = spawn here --> node::test;
            here -[apple]-> a;
            here -[apple]-> a;
            here -[banana]-> a;
            here -[banana]-> a;
            here -[banana]-> a;

            here !-[apple]-> a;
        }
    }
    """


edgey4 = \
    """
    node test;

    edge apple;
    edge banana;

    walker init {
        root {
            a = spawn here --> node::test;
            here --> a;
            here -[apple]-> a;
            here -[banana]-> a;

            here !-[generic]-> a;
        }
    }
    """

edgey5 = \
    """
    node test;

    edge apple;
    edge banana;

    walker init {
        root {
            a = spawn here --> node::test;
            here --> a;
            here --> a;
            here -[apple]-> a;
            here -[banana]-> a;

            here !-[generic]-> -[generic]->;
        }
    }
    """

edgey6 = \
    """
    node test;

    edge apple;
    edge banana;

    walker init {
        root {
            a = spawn here --> node::test;
            b = spawn here --> node::test;

            here -[apple]-> -[generic]->;
        }
    }
    """

edgey7 = \
    """
    node test;

    edge apple;
    edge banana;

    walker init {
        root {
            a = spawn here --> node::test;
            b = spawn here --> node::test;
            here --> a;
            here --> a;
            here -[apple]-> a;
            here -[apple]-> b;
            here -[banana]-> a;

            here !-[generic]-> -[apple]->;
        }
    }
    """

edge_access = \
    """
    node test;

    edge apple {
        has v1, v2;
    }

    edge banana {
        has x1, x2;
    }

    walker init {
        root {
            a = spawn here -[apple]-> node::test;
            b = spawn here -[banana]-> node::test;

            e = -[apple]->.edge[0];
            e.v1 = 7;
            e = --> node::test.edge[1];
            e.x1=8;
        }
    }
    """

has_assign = \
    """
    node test {
        has a=8;
    }


    walker init {
        root {
            a = spawn here --> node::test;
            b = spawn here --> node::test;

            std.log(a.a, b.a);
        }
    }
    """


set_get_global = \
    """
    walker setter {
        root {
            std.set_global('globby', 59);
        }
    }

    walker getter {
        has a;
        root {
            a=std.get_global('globby');
            std.log(std.get_global('globby'));
        }
    }
    """

set_get_global_dict = \
    """
    walker setter {
        root {
            std.set_global('globby',
            { "max_bot_count": 10, "max_ans_count": 100,
              "max_txn_count": 50000, "max_test_suite": 5,
              "max_test_cases": 50, "export_import": true,
              "analytics": true, "integration": "All"
            });
        }
    }

    walker getter {
        has a;
        root {
            a=std.get_global('globby');
            std.log(std.get_global('globby'));
            report std.get_global('globby');
        }
    }
    """

version_label = \
    """
    version: "alpha-1.0"

    walker setter {
        root {
            std.set_global('globby', 59);
        }
    }

    walker getter {
        has a;
        root {
            a=std.get_global('globby');
            std.log(std.get_global('globby'));
        }
    }
    """

sharable = \
    """
    node life {
    }

    walker init {
        root {
            new = spawn here --> node::life;
            take -->;
        }
        life {
            std.out(here);
        }
    }
    """

basic = \
    """
    node life {
    }

    walker init {
        root {
            new = spawn here --> node::life;
            take -->;
        }
        life {
        }
    }
    """

visibility_builtins = \
    """
    node test {
        has yo, mama;
    }

    edge apple {
        has v1, v2;
    }

    edge banana {
        has x1, x2;
    }

    walker init {
        root {
            a = spawn here -[apple]-> node::test;
            a.yo="Yeah i said";
            a.mama="Yo Mama Fool!";
            b = spawn here -[banana]-> node::test;

            e = -[apple]->.edge[0];
            e.v1 = 7;
            e = --> node::test.edge[1];
            e.x1=8;

            report [a.context, b.info, e.details];
        }
    }
    """

spawn_ctx_edge_node = \
    """
    node person: has name, age, birthday, profession;
    edge friend: has meeting_place;
    edge family: has kind;

    walker init {
        person1 = spawn here -[friend(meeting_place = "college")]->
            node::person(name = "Josh", age = 32);
        person2 = spawn here -[family(kind = "sister")] ->
            node::person(name = "Jane", age = 30);

        for i in -->{
            report i.context;
            report i.edge[0].context;
        }
    }
    """

filter_ctx_edge_node = \
    """
    node person: has name, age, birthday, profession;
    edge friend: has meeting_place;
    edge family: has kind;

    walker init {
        person1 = spawn here -[friend(meeting_place = "college")]->
            node::person(name = "Josh", age = 32);
        person2 = spawn here -[family(kind = "sister")] ->
            node::person(name = "Jane", age = 30);

        report --> node::person(name=='Jane')[0].context;
        report -[family(kind=="brother")]->;
    }
    """

null_handleing = \
    """
    node person: has name, age, birthday, profession;

    walker init {
        person1 = spawn here -->
            node::person(name = "Josh", age = 32);

        if(person1.birthday==null): report true;
        else: report false;

        if(person1.name==null): report true;
        else: report false;

        person1.name=null;
        report person1.name==null;
        person1.name=0;
        report person1.name==null;
    }
    """

bool_type_convert = \
    """
    node person: has name;

    walker init {
        p1 = spawn here -->
            node::person(name = "Josh");

        p1.name = true;
        report p1.name;
        std.log(p1.name);
        report p1.context;
    }
    """

typecasts = \
    """
    walker init {
        a=5.6;
        report (a+2);
        report (a+2).int;
        report (a+2).str;
        report (a+2).bool;
        report (a+2).int.float;

        if(a.str.type == str and !(a.int.type == str)
           and a.int.type == int):
            report "Types comes back correct";
    }
    """

typecasts_error = \
    """
    walker init {
        a=5.6;
        report (a+2);
        report (a+2).int;
        report (a+2).str;
        report (a+2).edge;
        report ("a+2").int.float;

        if(a.str.type == str and !(a.int.type == str)
           and a.int.type == int):
            report "Types comes back correct";
    }
    """

filter_on_context = \
    """
    node test {
        has yo, mama;
    }

    edge apple {
        has v1, v2;
    }

    edge banana {
        has x1, x2;
    }

    walker init {
        root {
            a = spawn here -[apple]-> node::test;
            a.yo="Yeah i said";
            a.mama="Yo Mama Fool!";
            b = spawn here -[banana]-> node::test;

            e = -[apple]->.edge[0];
            e.v1 = 7;
            e = --> node::test.edge[1];
            e.x1=8;

            report [a.context.{yo}, b.info.{jid,j_type}, e.details];
        }
    }
    """

string_manipulation = \
    """
    walker init {
        a=" tEsting me  ";
        report a[4];
        report a[4:7];
        report a[3:-1];
        report a.str::upper;
        report a.str::lower;
        report a.str::title;
        report a.str::capitalize;
        report a.str::swap_case;
        report a.str::is_alnum;
        report a.str::is_alpha;
        report a.str::is_digit;
        report a.str::is_title;
        report a.str::is_upper;
        report a.str::is_lower;
        report a.str::is_space;
        report a.str::count('t');
        report a.str::find('i');
        report a.str::split;
        report a.str::split('E');
        report a.str::startswith('tEs');
        report a.str::endswith('me');
        report a.str::replace('me', 'you');
        report a.str::strip;
        report a.str::strip(' t');
        report a.str::lstrip;
        report a.str::lstrip(' tE');
        report a.str::rstrip;
        report a.str::rstrip(' e');

        report a.str::upper.str::is_upper;
    }
    """

sub_list = \
    """
    walker init {
        a=[1,2,3,4,5,6,7,8,9];
        report a[4:7];
    }
    """

destroy_and_misc = \
    """
    node person: has name, age, birthday, profession;
    edge friend: has meeting_place;
    edge family: has kind;

    walker init {
        person1 = spawn here -[friend(meeting_place = "college")]->
            node::person(name = "Josh", age = 32);
        person2 = spawn here -[family(kind = "sister")] ->
            node::person(name = "Jane", age = 30);

        report person1.name;
        destroy person1.name;
        report person1.context;
        person1.name="pete";
        report person1.context;
        a=[1,2,3];
        destroy a[1];
        report a;
        b={'a': 'b', 'c':'d'};
        destroy b['c'];
        report b;
        a=[1,2,3,5,6,7,8,9];
        destroy a[2:4];
        report a;
        a[2:4]=[45,33];
        report a;
        destroy a;
        report a;
        person1.banana=45;
        report person1.context;
        report 'age' in person1.context;
    }
    """

arbitrary_assign_on_element = \
    """
    node person: has name, age, birthday, profession;
    walker init {
        some = spawn here --> node::person;
        some.apple = 45;
        report some.context;
    }
    """

try_else_stmts = \
    """
    walker init {
        a=null;
        try {a=2/0;}
        else with err {report err;}
        try {a=2/0;}
        else {report 'dont need err';}
        try {a=2/0;}
        try {a=2/0;}
        report a;
        try {a=2/1;}
        report a;
    }
    """
