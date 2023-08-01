create table si.check_for_speaker_diraization
(
    id           int auto_increment
        primary key,
    record_id    varchar(32)  null,
    wav_duration float        null,
    file_url     varchar(255) null,
    selected_url varchar(255) null,
    asr_text     text         null,
    selected_times     text         null,
    create_time  datetime     null,
    constraint check_for_speaker_diraization_pk
        unique (record_id)
);

